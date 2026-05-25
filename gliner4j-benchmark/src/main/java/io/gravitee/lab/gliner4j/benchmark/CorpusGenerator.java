/*
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
package io.gravitee.lab.gliner4j.benchmark;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import net.datafaker.Faker;

/**
 * Generates fake benchmark corpora using {@link Faker}.
 *
 * <p>The output is fully synthetic — names, companies, addresses, emails, IBANs, card numbers
 * and so on come from datafaker's randomised generators. With a fixed seed the same sample
 * set is produced every run, so different benchmark invocations stay comparable.
 *
 * <p>Profiles:
 * <ul>
 *   <li><b>base</b> — news/encyclopedia-flavoured sentences exposing person, organization,
 *       location, date, event and product entities.</li>
 *   <li><b>pii</b> — record-flavoured sentences exposing email, phone, IBAN, card_number,
 *       ip_address, password, address and so on.</li>
 * </ul>
 *
 * <p>Each generated sample is padded with additional sentences of the same flavour until it
 * crosses an approximate character target (≈ 4 chars per token).
 */
final class CorpusGenerator {

  private static final int CHARS_PER_TOKEN = 4;

  private final Faker faker;
  private final Random rng;

  CorpusGenerator(long seed) {
    this.rng = new Random(seed);
    this.faker = new Faker(rng);
  }

  /**
   * Build {@code count} samples targeting roughly {@code targetTokens} tokens each.
   *
   * @param profile      either {@code "base"} or {@code "pii"}
   * @param targetTokens approximate token length per sample (16 / 64 / 256 / 512)
   * @param count        number of samples to produce
   */
  List<String> generate(String profile, int targetTokens, int count) {
    var targetChars = targetTokens * CHARS_PER_TOKEN;
    var out = new ArrayList<String>(count);
    for (int i = 0; i < count; i++) {
      out.add(buildSample(profile, targetChars));
    }
    return out;
  }

  private String buildSample(String profile, int targetChars) {
    var sb = new StringBuilder(targetChars + 64);
    while (sb.length() < targetChars) {
      if (!sb.isEmpty()) sb.append(' ');
      sb.append("base".equals(profile) ? baseSentence() : piiSentence());
    }
    return sb.toString();
  }

  // ── base profile templates ─────────────────────────────────────────────────

  private String baseSentence() {
    return switch (rng.nextInt(8)) {
      case 0 -> "%s unveiled the %s at %s's %s event in %s on %s.".formatted(
        person(),
        product(),
        company(),
        event(),
        city(),
        date()
      );
      case 1 -> "The %s board met in %s on %s to review the acquisition of %s.".formatted(
        company(),
        city(),
        date(),
        company()
      );
      case 2 -> "%s, chief executive of %s, opened the %s in %s.".formatted(
        person(),
        company(),
        event(),
        city()
      );
      case 3 -> "%s acquired %s for %d million units in a deal signed in %s on %s.".formatted(
        company(),
        company(),
        100 + rng.nextInt(900),
        city(),
        date()
      );
      case 4 -> "Researchers at %s in %s published a study about %s in %s.".formatted(
        company(),
        city(),
        product(),
        date()
      );
      case 5 -> "%s scored twice for %s against %s at the stadium in %s on %s.".formatted(
        person(),
        company(),
        company(),
        city(),
        date()
      );
      case 6 -> "The central bank kept rates steady at the policy meeting chaired by %s in %s on %s.".formatted(
        person(),
        city(),
        date()
      );
      case 7 -> "%s announced the %s product line at the %s in %s last %s.".formatted(
        company(),
        product(),
        event(),
        city(),
        date()
      );
      default -> "%s visited %s in %s on %s.".formatted(
        person(),
        company(),
        city(),
        date()
      );
    };
  }

  // ── pii profile templates ──────────────────────────────────────────────────

  private String piiSentence() {
    return switch (rng.nextInt(10)) {
      case 0 -> "Contact %s at %s or call %s.".formatted(
        person(),
        email(),
        phone()
      );
      case 1 -> "Charge card %s expiry %s cvv %s to invoice %d.".formatted(
        card(),
        cardExpiry(),
        cvv(),
        1000 + rng.nextInt(9000)
      );
      case 2 -> "Login was %s with password %s from %s.".formatted(
        username(),
        password(),
        ipv4()
      );
      case 3 -> "Wire to IBAN %s reference %s for %s.".formatted(
        iban(),
        invoiceRef(),
        company()
      );
      case 4 -> "Account holder %s lives at %s in %s.".formatted(
        person(),
        streetAddress(),
        city()
      );
      case 5 -> "SSN %s belongs to %s born on %s.".formatted(
        ssn(),
        person(),
        date()
      );
      case 6 -> "API key %s issued to %s on %s expires after the next quarter.".formatted(
        apiKey(),
        email(),
        date()
      );
      case 7 -> "Driver license %s expires on %s for user %s.".formatted(
        license(),
        date(),
        person()
      );
      case 8 -> "Bank account %s routing %s owned by %s with phone %s.".formatted(
        bankAccount(),
        routing(),
        person(),
        phone()
      );
      case 9 -> "Recovery code %s sent to %s for account holder %s.".formatted(
        recoveryCode(),
        email(),
        person()
      );
      default -> "Connecting from %s with bearer token %s on %s.".formatted(
        ipv4(),
        accessToken(),
        date()
      );
    };
  }

  // ── helper accessors over Faker ────────────────────────────────────────────

  private String person() {
    return faker.name().fullName();
  }

  private String company() {
    return faker.company().name();
  }

  private String city() {
    return faker.address().cityName();
  }

  private String product() {
    return faker.commerce().productName();
  }

  private String event() {
    return faker.book().title();
  }

  private String date() {
    return "%04d-%02d-%02d".formatted(
      2030 + rng.nextInt(15),
      1 + rng.nextInt(12),
      1 + rng.nextInt(28)
    );
  }

  private String email() {
    return faker.internet().emailAddress();
  }

  private String phone() {
    return faker.phoneNumber().phoneNumber();
  }

  private String card() {
    return faker.finance().creditCard();
  }

  private String cardExpiry() {
    return "%02d/%02d".formatted(1 + rng.nextInt(12), rng.nextInt(99));
  }

  private String cvv() {
    return "%03d".formatted(rng.nextInt(1000));
  }

  private String iban() {
    return faker.finance().iban();
  }

  private String invoiceRef() {
    return "INV-%05d".formatted(rng.nextInt(100000));
  }

  private String streetAddress() {
    return faker.address().streetAddress();
  }

  private String ssn() {
    return faker.idNumber().ssnValid();
  }

  private String username() {
    return faker.internet().username();
  }

  private String password() {
    return faker.internet().password(8, 16, true, true, true);
  }

  private String ipv4() {
    return faker.internet().ipV4Address();
  }

  private String apiKey() {
    return "sk-live-" + faker.regexify("[a-z0-9]{16}");
  }

  private String license() {
    return faker.regexify("[A-Z][0-9]{4}-[0-9]{5}-[0-9]{5}");
  }

  private String bankAccount() {
    return faker.regexify("[0-9]{12}");
  }

  private String routing() {
    return faker.regexify("[0-9]{9}");
  }

  private String recoveryCode() {
    return faker.regexify("[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}");
  }

  private String accessToken() {
    return "Bearer " + faker.regexify("[A-Za-z0-9]{20}");
  }
}
