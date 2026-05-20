#
# Copyright © 2015 The Gravitee team (http://gravitee.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# GLiNER4j JMH benchmark runner.
#
# Builds the benchmark uber-jar and bakes it into a small JDK image. Model
# files are NOT baked in — mount them at run time:
#
#     docker run --rm \
#         --cpuset-cpus=0-3 --cpu-shares=1024 \
#         -v $PWD/models:/app/models \
#         -v $PWD/results:/app/results \
#         gliner4j-benchmark:latest \
#         -p profile=pii -p batchSize=4 -p entityCount=42 \
#         -rf json -rff /app/results/bench.json
#
# Build (run from the repo root so the context can see all modules):
#     docker build -f gliner4j-benchmark/Dockerfile -t gliner4j-benchmark:latest .

# ── Build stage ────────────────────────────────────────────────────────────
FROM maven:3.9-eclipse-temurin-21 AS build

WORKDIR /workspace

# Copy poms first to leverage Docker layer caching for dependency resolution.
COPY pom.xml ./
COPY gliner4j-core/pom.xml gliner4j-core/
COPY gliner4j-demo/pom.xml gliner4j-demo/
COPY gliner4j-benchmark/pom.xml gliner4j-benchmark/
RUN mvn -B -q -pl gliner4j-benchmark -am dependency:go-offline -DskipTests || true

# Copy the rest and build the benchmark uber-jar.
COPY lombok.config ./
COPY gliner4j-core gliner4j-core
COPY gliner4j-demo gliner4j-demo
COPY gliner4j-benchmark gliner4j-benchmark
RUN mvn -B -q -pl gliner4j-benchmark -am package -DskipTests

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM eclipse-temurin:21-jre

WORKDIR /app

COPY --from=build /workspace/gliner4j-benchmark/target/gliner4j-benchmark.jar /app/gliner4j-benchmark.jar

# Default: print JMH help. Override at `docker run` with any JMH args.
ENTRYPOINT ["java", "-jar", "/app/gliner4j-benchmark.jar"]
CMD ["-h"]
