/**
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// The "DEBERTA" backend: a ggml backend plugin (ggml_backend_load) in the shape of ggml-blas.cpp.
// It owns no buffers — its device reports the stock CUDA backend's buffer type and accepts it in
// supports_buft, so ggml_backend_sched schedules our node between CUDA splits without copies —
// and computes exactly one op: GGML_OP_CUSTOM nodes whose function pointer is our marker.

#include "ggml-deberta.h"
#include "deberta-attn.h"
#include "lstm.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <string>

#define GLINER4J_STR2(x) #x
#define GLINER4J_STR(x) GLINER4J_STR2(x)
#define GLINER4J_DEBERTA_VERSION_STRING "gliner4j-deberta/1 ggml-backend-api/" GLINER4J_STR(GGML_BACKEND_API_VERSION)

namespace {

struct deberta_backend_context {
    int          device;
    cudaStream_t stream;
    bool         reference;
    // Host-side synchronisation around graph_compute (default). Off when the scheduler orders
    // splits with events (patched ggml + sched created with parallel=true), see
    // gliner4j_deberta_set_host_sync.
    bool         host_sync;
};

// global switch, set by the host once it knows how the scheduler was created
static bool g_host_sync = true;

bool deberta_qkv_f32(void) {
    return true;   // feature marker: the attention op takes f32 q / k / v (see GLINER4J_DEBERTA_QKV_F32)
}

void deberta_set_host_sync(bool on) {
    g_host_sync = on;
}

struct deberta_device_context {
    int device;
    std::string name;
    std::string description;
};

// ---- the op marker ---------------------------------------------------------------------------
// ggml_custom_4d stores this pointer in op_params; supports_op matches on it. The CPU backend
// accepts every op and executes GGML_OP_CUSTOM, so if scheduling ever goes wrong this body runs on
// the host with device pointers — abort loudly instead.
void deberta_attn_marker(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(dst); GGML_UNUSED(ith); GGML_UNUSED(nth); GGML_UNUSED(userdata);
    GGML_ABORT("gliner4j deberta attention was scheduled on the CPU backend (plugin device missing from the scheduler?)");
}

void lstm_marker(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(dst); GGML_UNUSED(ith); GGML_UNUSED(nth); GGML_UNUSED(userdata);
    GGML_ABORT("gliner4j lstm was scheduled on the CPU backend (plugin device missing from the scheduler?)");
}

const char * deberta_version(void) {
    return GLINER4J_DEBERTA_VERSION_STRING;
}

bool is_our_node(const struct ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) {
        return false;
    }
    struct ggml_custom_op_params params;
    std::memcpy(&params, op->op_params, sizeof(params));
    return params.fun == deberta_attn_marker || params.fun == lstm_marker;
}

ggml_custom_op_t custom_fn(const struct ggml_tensor * op) {
    struct ggml_custom_op_params params;
    std::memcpy(&params, op->op_params, sizeof(params));
    return params.fun;
}

// ---- CUDA buffer type via the registry (libggml-cuda is a MODULE, never linked) ----------------
ggml_backend_buffer_type_t cuda_buffer_type(int device) {
    ggml_backend_reg_t cuda = ggml_backend_reg_by_name("CUDA");
    if (cuda == nullptr) {
        return nullptr;
    }
    if ((size_t) device >= ggml_backend_reg_dev_count(cuda)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(cuda, device));
}

// ---- backend ---------------------------------------------------------------------------------
const char * backend_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "DEBERTA";
}

void backend_free(ggml_backend_t backend) {
    auto * ctx = (deberta_backend_context *) backend->context;
    cudaStreamDestroy(ctx->stream);
    delete ctx;
    delete backend;
}

void backend_synchronize(ggml_backend_t backend) {
    auto * ctx = (deberta_backend_context *) backend->context;
    cudaError_t err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) {
        GGML_ABORT("gliner4j deberta: cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
    }
}

enum ggml_status backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (deberta_backend_context *) backend->context;
    cudaSetDevice(ctx->device);
    // Our operands were produced asynchronously by the CUDA backend on a stream we cannot see.
    // ggml_backend_sched only synchronises the previous backend when a split has no inputs; ours
    // always has one (`lens`, a user input copied in), so it synchronises *us* instead. Wait for
    // the whole device before reading.
    cudaError_t sync = g_host_sync ? cudaDeviceSynchronize() : cudaSuccess;
    if (sync != cudaSuccess) {
        GGML_LOG_ERROR("gliner4j deberta: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(sync));
        return GGML_STATUS_FAILED;
    }
    const int n_nodes = ggml_graph_n_nodes(cgraph);
    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(cgraph, i);
        switch (node->op) {
            case GGML_OP_CUSTOM: {
                if (!is_our_node(node)) {
                    GGML_ABORT("gliner4j deberta: unknown custom op %s", node->name);
                }
                cudaError_t err = custom_fn(node) == lstm_marker
                    ? gliner4j_lstm_launch(node, ctx->stream)
                    : gliner4j_deberta_attn_launch(node, ctx->stream, ctx->reference);
                if (err != cudaSuccess) {
                    GGML_LOG_ERROR("gliner4j deberta: kernel launch failed: %s\n", cudaGetErrorString(err));
                    return GGML_STATUS_FAILED;
                }
                break;
            }
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("gliner4j deberta: unsupported op %s", ggml_op_desc(node));
        }
    }
    // The same asymmetry on the way out: the scheduler waits for us only when the next CUDA split
    // has no inputs, and the ones that carry a user input (the task heads) would read our output
    // while the kernel is still running. Finish on the host before handing back.
    cudaError_t done = g_host_sync ? cudaStreamSynchronize(ctx->stream) : cudaSuccess;
    if (done != cudaSuccess) {
        GGML_LOG_ERROR("gliner4j deberta: kernel failed: %s\n", cudaGetErrorString(done));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

// ---- events: plain cudaEvent_t in event->context, the same shape the CUDA backend uses, so
// either backend can wait on the other's events on its own stream ----------------------------
void backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * ctx = (deberta_backend_context *) backend->context;
    cudaError_t err = cudaEventRecord((cudaEvent_t) event->context, ctx->stream);
    if (err != cudaSuccess) {
        GGML_ABORT("gliner4j deberta: cudaEventRecord failed: %s", cudaGetErrorString(err));
    }
}

void backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * ctx = (deberta_backend_context *) backend->context;
    cudaError_t err = cudaStreamWaitEvent(ctx->stream, (cudaEvent_t) event->context, 0);
    if (err != cudaSuccess) {
        GGML_ABORT("gliner4j deberta: cudaStreamWaitEvent failed: %s", cudaGetErrorString(err));
    }
}

struct ggml_backend_i backend_iface = {
    /* .get_name                = */ backend_get_name,
    /* .free                    = */ backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ backend_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ backend_graph_compute,
    /* .event_record            = */ backend_event_record,
    /* .event_wait              = */ backend_event_wait,
    /* .graph_optimize          = */ nullptr,
};

ggml_guid_t backend_guid(void) {
    static ggml_guid guid = { 0x67, 0x6c, 0x69, 0x6e, 0x65, 0x72, 0x34, 0x6a, 0x64, 0x65, 0x62, 0x65, 0x72, 0x74, 0x61, 0x01 };
    return &guid;
}

// ---- device ----------------------------------------------------------------------------------
const char * device_get_name(ggml_backend_dev_t dev) {
    return ((deberta_device_context *) dev->context)->name.c_str();
}

const char * device_get_description(ggml_backend_dev_t dev) {
    return ((deberta_device_context *) dev->context)->description.c_str();
}

void device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * ctx = (deberta_device_context *) dev->context;
    *free = 0;
    *total = 0;
    if (cudaSetDevice(ctx->device) == cudaSuccess) {
        cudaMemGetInfo(free, total);
    }
}

enum ggml_backend_dev_type device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    // ACCEL, not GPU: GgmlWeights picks its weights device with ggml_backend_dev_by_type(GPU) and
    // must keep finding the stock CUDA device there.
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

void device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = device_get_name(dev);
    props->description = device_get_description(dev);
    props->type        = device_get_type(dev);
    device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ false,
    };
}

ggml_backend_t device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * dctx = (deberta_device_context *) dev->context;
    if (cuda_buffer_type(dctx->device) == nullptr) {
        GGML_LOG_ERROR("gliner4j deberta: the CUDA backend is not registered, cannot initialise\n");
        return nullptr;
    }
    if (cudaSetDevice(dctx->device) != cudaSuccess) {
        GGML_LOG_ERROR("gliner4j deberta: cudaSetDevice(%d) failed\n", dctx->device);
        return nullptr;
    }
    auto * ctx = new deberta_backend_context;
    ctx->device = dctx->device;
    ctx->reference = std::getenv("GLINER4J_DEBERTA_REF") != nullptr;
    ctx->host_sync = true;
    cudaError_t err = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("gliner4j deberta: cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        delete ctx;
        return nullptr;
    }
    return new ggml_backend {
        /* .guid    = */ backend_guid(),
        /* .iface   = */ backend_iface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
}

ggml_backend_event_t device_event_new(ggml_backend_dev_t dev) {
    auto * dctx = (deberta_device_context *) dev->context;
    cudaSetDevice(dctx->device);
    cudaEvent_t ev;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        return nullptr;
    }
    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

void device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    cudaEventDestroy((cudaEvent_t) event->context);
    delete event;
}

void device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    cudaError_t err = cudaEventSynchronize((cudaEvent_t) event->context);
    if (err != cudaSuccess) {
        GGML_ABORT("gliner4j deberta: cudaEventSynchronize failed: %s", cudaGetErrorString(err));
    }
}

ggml_backend_buffer_type_t device_get_buffer_type(ggml_backend_dev_t dev) {
    return cuda_buffer_type(((deberta_device_context *) dev->context)->device);
}

ggml_backend_buffer_t device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev); GGML_UNUSED(ptr); GGML_UNUSED(size); GGML_UNUSED(max_tensor_size);
    return nullptr;
}

bool device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    // Not GGML_OP_NONE: leaves (weights) must stay assigned to the CUDA backend in the scheduler.
    switch (op->op) {
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_CUSTOM:
            return is_our_node(op);
        default:
            return false;
    }
}

bool device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_buffer_type_t cuda = cuda_buffer_type(((deberta_device_context *) dev->context)->device);
    return cuda != nullptr && buft == cuda;
}

const struct ggml_backend_device_i device_iface = {
    /* .get_name             = */ device_get_name,
    /* .get_description      = */ device_get_description,
    /* .get_memory           = */ device_get_memory,
    /* .get_type             = */ device_get_type,
    /* .get_props            = */ device_get_props,
    /* .init_backend         = */ device_init_backend,
    /* .get_buffer_type      = */ device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ device_buffer_from_host_ptr,
    /* .supports_op          = */ device_supports_op,
    /* .supports_buft        = */ device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ device_event_new,
    /* .event_free           = */ device_event_free,
    /* .event_synchronize    = */ device_event_synchronize,
};

// ---- registry --------------------------------------------------------------------------------
const char * reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "DEBERTA";
}

size_t reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

ggml_backend_dev_t reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static deberta_device_context dctx = {
        /* .device      = */ std::getenv("GLINER4J_DEBERTA_DEVICE") ? std::atoi(std::getenv("GLINER4J_DEBERTA_DEVICE")) : 0,
        /* .name        = */ "DEBERTA0",
        /* .description = */ "gliner4j fused DeBERTa disentangled attention (CUDA)",
    };
    static ggml_backend_device device = {
        /* .iface   = */ device_iface,
        /* .reg     = */ reg,
        /* .context = */ &dctx,
    };
    return &device;
}

void * reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, GLINER4J_DEBERTA_ATTN_FN) == 0) {
        return (void *) deberta_attn_marker;
    }
    if (std::strcmp(name, GLINER4J_LSTM_FN) == 0) {
        return (void *) lstm_marker;
    }
    if (std::strcmp(name, GLINER4J_DEBERTA_VERSION) == 0) {
        return (void *) deberta_version;
    }
    if (std::strcmp(name, GLINER4J_DEBERTA_SET_HOST_SYNC) == 0) {
        return (void *) deberta_set_host_sync;
    }
    if (std::strcmp(name, GLINER4J_DEBERTA_KERNEL_REPORT) == 0) {
        return (void *) gliner4j_deberta_attn_report;
    }
    if (std::strcmp(name, GLINER4J_DEBERTA_QKV_F32) == 0) {
        return (void *) deberta_qkv_f32;
    }
    return nullptr;
}

const struct ggml_backend_reg_i reg_iface = {
    /* .get_name         = */ reg_get_name,
    /* .get_device_count = */ reg_get_device_count,
    /* .get_device       = */ reg_get_device,
    /* .get_proc_address = */ reg_get_proc_address,
};

} // namespace

ggml_backend_reg_t ggml_backend_deberta_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ reg_iface,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_deberta_reg)
