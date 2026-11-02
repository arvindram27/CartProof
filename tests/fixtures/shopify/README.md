Shopify fixture set (P40). PROVENANCE AND LIMITS - READ BEFORE CITING:
- cart_create.json and cart_query.json are ADAPTED from the OFFICIAL
  EXAMPLE responses in the shopify.dev Storefront API cart guide
  (https://shopify.dev/docs/storefronts/headless/building-with-the-storefront-api/cart/manage),
  captured 2026-11-02. Adaptation, marked honestly: the published examples
  omit pageInfo; "pageInfo": {"hasNextPage": false} was ADDED to the lines
  object because the backend requires an explicit completeness signal and
  treats its absence as malformed. All other fields are verbatim. They are
  Shopify's published examples, NOT live responses from
  cartproof.myshopify.com.
- cart_lines_add.json is CONSTRUCTED in the documented CartLinesAddPayload
  shape (cart / userErrors / warnings per
  https://shopify.dev/docs/api/storefront/latest/mutations/cartLinesAdd)
  reusing the same published example values.
- mapping.json maps synthetic test SKUs to the published example variant
  GID. It is NOT the real store's variant mapping; real variant IDs are
  captured in a future Stage 1 live smoke, which would require
  separate approval (none granted).
- The "?key=examplekey1234567890" value is Shopify's own published
  placeholder, not a secret. Real cart IDs never appear in this repo.
- Behavioral tests do NOT rely on these static files alone:
  tests/shopify_fake.py is a STATEFUL in-memory simulation whose merge
  behavior (cartLinesAdd increments an existing line) is an explicit
  ASSUMPTION - P39 remote unknown #2 - pinned only by the Stage 1 live
  smoke. The canonicalizer is tested against duplicate-line carts, so the
  code is correct under either real behavior.
These fixtures settle CODE behavior only. Live unknowns (P39 design
section 6) remain open until Stage 1.
