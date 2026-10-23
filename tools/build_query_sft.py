#!/usr/bin/env python3
"""
build_query_sft.py

Builds the data for a natural-language order search: a request such as
  "show me unpaid electronics orders over $500 from last month, newest first"
becomes a command-line style reply that code turns into a query:
  orders find --status unpaid --category electronics --min-amount 500 --when last-month --sort newest

The reply is written like a command so internal/constrain can restrict decoding to it. Dates are
symbolic (--when last-month, --last-days 30, --month march --year 2025): the model never does
calendar arithmetic, the application resolves them against today's date.

Requests also come with typos and dropped apostrophes (the reply is unaffected), status nouns ("refund count for
March"), "revenue" phrasing, a count/total word at the end ("pending orders count"), "last 3 orders", short
conversational forms ("what's outstanding", "only electronics") and many refusals of actions and of imperative
requests that are not order searches ("ship order 90", "play some music").

Besides the filters it knows order ids (--id), open date ranges (--before, --after) and search-box style
requests ("unpaid", "over 500", "oct 15", "customer Globex", "all").

Subcommands: find (list orders), count ("how many ..."), total ("total revenue of ..."). Requests that are
not an order search (small talk, other topics, actions such as "delete order 5") get "orders unknown".
Options come in a fixed order, so one request has one reply.

Outputs (in --out-dir):
  query_sft.json       training pairs
  query_heldout.json   held-out pairs with a "split": "seen_combo" (new values and wording of trained
                       status/category combinations), "unseen_combo" (combinations never trained) or
                       "out_of_scope" (wordings of non-search requests that are never trained)
  query_tasks.json     benchmark probes for `mignon eval -benchmark`

Usage: python3 tools/build_query_sft.py --out-dir data/exp/query [--train 60000]
"""

import argparse
import collections
import json
import os

import probes
from techcorpus_common import save_json, seeded_rng, stable_hash

STATUSES = {
    "paid": (["paid", "settled", "cleared", "fully paid", "paid-up"],
             ["that are paid", "which have been paid", "that were paid", "we got paid for", "that the customer paid for",
              "that have been settled", "where the payment came through", "that are paid up"]),
    "unpaid": (["unpaid", "outstanding", "open", "unsettled", "not yet paid"],
               ["that haven't been paid", "not paid yet", "that are still unpaid", "waiting for payment", "that nobody has paid for",
                "that still need to be paid", "we haven't been paid for", "with money still owed", "where the customer still owes us",
                "awaiting payment", "that are still open"]),
    "overdue": (["overdue", "late", "past due", "delinquent"],
                ["that are overdue", "that are running late", "that are past due", "that are late", "that are way past their due date",
                 "that should have been paid by now", "whose payment is late", "that missed their payment deadline"]),
    "refunded": (["refunded", "returned", "sent-back"],
                 ["that were refunded", "we refunded", "did we refund", "that got refunded", "with a refund", "that customers sent back",
                  "where we gave the money back", "that were returned", "that came back to us", "that got a refund"]),
    "cancelled": (["cancelled", "canceled", "voided", "called-off"],
                  ["that were cancelled", "we cancelled", "did we cancel", "customers cancelled", "that got cancelled", "that were canceled",
                   "that fell through", "that were called off", "that never went ahead", "that were dropped by the customer",
                   "that got voided"]),
    "shipped": (["shipped", "dispatched", "sent-out", "in-transit"],
                ["that have shipped", "we shipped", "did we ship", "we sent out", "that were shipped", "that are on their way",
                 "that have left the warehouse", "that have gone out", "that are already out the door", "that are in transit",
                 "that went out"]),
    "pending": (["pending", "unprocessed", "not-yet-processed"],
                ["that are pending", "that are still pending", "waiting to be processed", "that haven't been processed yet",
                 "we haven't touched yet", "that are still waiting to go out", "that are awaiting processing",
                 "that are still in the queue"]),
}
CATEGORIES = {
    "electronics": ["in electronics", "from the electronics category", "for electronics"],
    "books": ["in books", "from the books category", "for books"],
    "clothing": ["in clothing", "from the clothing category", "for clothing"],
    "home": ["in home goods", "from the home category", "for home"],
    "toys": ["in toys", "from the toys category", "for toys"],
    "sports": ["in sports", "from the sports category", "for sports"],
    "beauty": ["in beauty", "from the beauty category", "for beauty"],
    "groceries": ["in groceries", "from the groceries category", "for groceries"],
}
CATEGORY_ADJ = {c: c for c in CATEGORIES}
# other ways to name a category; the plain name stays the most common
CATEGORY_WORDS = {
    "electronics": ["gadgets", "tech", "electronic", "electronic goods", "devices"],
    "books": ["book", "reading material"],
    "clothing": ["apparel", "clothes", "fashion", "garment"],
    "home": ["household", "homeware", "home and garden", "furnishing"],
    "toys": ["toy", "games and toys", "kids' toys"],
    "sports": ["sporting goods", "sports gear", "athletic", "sports equipment"],
    "beauty": ["cosmetics", "makeup", "skincare", "beauty products"],
    "groceries": ["grocery", "food", "supermarket", "food and drink"],
}
CATEGORY_FRAMES = ["in {w}", "from the {w} category", "for {w}", "in the {w} department", "from {w}", "of {w}",
                   "{w} related", "in the {w} section"]


def category_word(rng, cat, plain=0.55):
    return cat if rng.random() < plain else rng.choice(CATEGORY_WORDS[cat])


def category_clause(rng, cat):
    """A clause after "orders": the original phrasings, or a frame with a synonym."""
    if rng.random() < 0.5:
        return rng.choice(CATEGORIES[cat])
    return rng.choice(CATEGORY_FRAMES).format(w=category_word(rng, cat))
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
YEARS = [str(y) for y in range(2018, 2028)]
MONTH_ABBR = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_DAYS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def amount(rng):
    """Any whole number a person might type: round figures are common, odd ones (730, 1350, 12000) too.
    A short fixed list would be memorised and the model would answer 700 for 800."""
    r = rng.random()
    if r < 0.3:
        return rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 750,
                           800, 1000, 1200, 1500, 2000, 2500, 3000, 5000, 10000])
    if r < 0.6:
        return rng.randint(10, 999)
    if r < 0.85:
        n = rng.randint(1000, 20000)
        return n - n % 10 if rng.random() < 0.5 else n
    return rng.choice([rng.randint(1, 9), rng.randint(20000, 99999)])


def small_count(rng):
    """A limit or a number of days: small numbers, any of them."""
    return rng.choice([rng.randint(1, 30), rng.randint(1, 30), rng.randint(2, 100), rng.choice([30, 60, 90, 365])])


def ordinal(n):
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def date_text(rng, month, day, year):
    """A calendar day as a person writes it: "Oct 15", "15th of March 2025", "2025-03-15"."""
    mi = MONTHS.index(month)
    if year and rng.random() < 0.1:
        return f"{year}-{mi + 1:02d}-{day:02d}"
    name = month.capitalize() if rng.random() < 0.8 else month
    if rng.random() < 0.4:
        name = MONTH_ABBR[mi].capitalize() + ("t" if MONTH_ABBR[mi] == "sep" and rng.random() < 0.5 else "")
    forms = [f"{name} {day}", f"{name} {ordinal(day)}", f"{day} {name}", f"{ordinal(day)} of {name}",
             f"the {ordinal(day)} of {name}", f"{name} {day}"]
    text = rng.choice(forms)
    if year:
        text += rng.choice([" ", ", "]) + str(year)
    return text
RELATIVE = {
    "today": ["today", "from today", "placed today", "so far today", "earlier today", "that came in today"],
    "yesterday": ["yesterday", "from yesterday", "placed yesterday", "the day before today", "that came in yesterday",
                  "during the previous day"],
    "this-week": ["this week", "from this week", "placed this week", "so far this week", "during the current week",
                  "that came in this week"],
    "last-week": ["last week", "from last week", "placed last week", "during the previous week", "in the week before this one",
                  "that came in last week"],
    "this-month": ["this month", "from this month", "placed this month", "so far this month", "in the current month",
                   "that came in this month"],
    "last-month": ["last month", "from last month", "placed last month", "during the previous month",
                   "in the month before this one", "that came in last month"],
    "this-year": ["this year", "from this year", "placed this year", "so far this year", "year to date", "in the current year"],
    "last-year": ["last year", "from last year", "placed last year", "during the previous year", "in the year before this one",
                  "that came in last year"],
}
SORTS = {
    "newest": ["newest first", "most recent first", "latest first", "sorted by date, newest first"],
    "oldest": ["oldest first", "earliest first", "sorted by date, oldest first"],
    "highest": ["biggest first", "highest amount first", "largest first", "sorted by amount, highest first",
                "by amount descending", "most expensive first", "priciest first"],
    "lowest": ["smallest first", "lowest amount first", "cheapest first", "sorted by amount, lowest first",
               "by amount ascending", "least expensive first"],
}
FIRST = ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Stark", "Wayne", "Wonka", "Soylent", "Cyberdyne",
         "Tyrell", "Vandelay", "Pied", "Dunder", "Oscorp", "Aperture", "Nakatomi", "Bluth", "Gringotts", "Monarch",
         "Alice", "Bob", "Carla", "Dmitri", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jamal", "Kofi", "Lena",
         "Marco", "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tariq"]
# Real names of many origins and lengths: copying a long unusual name letter by letter is what the model finds hardest.
WORLD_NAMES = ["Nikos", "Eleni", "Giorgos", "Dimitra", "Konstantinos", "Alexandros", "Katerina", "Stavros", "Theodora",
               "Aristotelis", "Yannis", "Sofia", "Mateusz", "Agnieszka", "Wojciech", "Oluwaseun", "Chidinma", "Babatunde",
               "Siddharth", "Aishwarya", "Venkatesh", "Mohammed", "Fatima", "Abdelrahman", "Yusuf", "Leila", "Chen",
               "Xiaoming", "Yuki", "Haruto", "Minjun", "Seo-yeon", "Nguyen", "Tuan", "Bartholomew", "Christopher",
               "Elizabeth", "Margaret", "Sebastian", "Valentina", "Guillermo", "Alejandro", "Esperanza", "Francesca",
               "Giovanni", "Lorenzo", "Ingrid", "Bjorn", "Magnus", "Siobhan", "Declan", "Anastasia", "Vladimir",
               "Svetlana", "Ekaterina", "Zsofia", "Emeka", "Thandiwe", "Kwame", "Amara", "Ravi", "Anjali", "Jonas",
               "Maximilian", "Charlotte", "Theodore", "Penelope", "Evangelos", "Panagiotis", "Charalambos", "Themistoklis",
               "Marilena", "Dionysia", "Kleopatra", "Hippolyta", "Thomas", "Maria", "Ahmed", "Olga", "Ivan", "Zoe"]
LAST = ["Corp", "Ltd", "Industries", "Labs", "Group", "Smith", "Okafor", "Nguyen", "Garcia", "Rossi", "Kim",
        "Novak", "Silva", "Haddad", "Larsen", "Moreau", "Tanaka", "Ivanov", "Costa", "Weber", "Papadopoulos", "Karagiannis",
        "Nowak", "Adeyemi", "Kowalski", "Fitzgerald", "Mukherjee", "Hashemi", "Johansson", "Oyelaran", "Vasquez"]

LEADS = {
    "find": ["show me", "find", "list", "get", "give me", "I need", "pull up", "display", "can you find",
             "search for", "look up", "show", "fetch", "what are the", "let me see the", "any", "",
             "I'd like to see", "I was wondering if you could show me", "could you please pull up", "I want to look at",
             "do we have any", "is there any", "tell me about", "hey, show me", "can I see", "I'm after", "bring up",
             "I'm looking for", "put together a list of", "give me a rundown of", "let me know about"],
    "count": ["how many", "count the", "count", "number of", "the number of", "what's the number of",
              "tell me how many", "how many of our", "tally up the", "count of", "give me a count of",
              "do we have any, and how many", "", "quick question: how many", "I'm curious how many", "any idea how many",
              "let me know how many", "give me the numbers on", "what's the count of", "could you tell me how many",
              "I need to know how many", "what's the tally of"],
    "total": ["total of", "what's the total of", "total value of", "sum up", "what is the total amount of",
              "how much revenue from", "revenue from", "add up", "grand total of", "sum of", "total",
              "what's the total for", "how much have we made from", "how much did we make from", "",
              "how much cash did we bring in from", "what did we earn from", "what's the take from", "income from",
              "what's our income from", "how much money came in from", "what are the takings from",
              "I need the total for"],
}
NOUNS = {
    "find": ["orders", "orders", "orders", "invoices", "purchases", "transactions"],
    "count": ["orders", "orders", "orders", "invoices", "purchases", "transactions"],
    "total": ["orders", "orders", "invoices", "purchases", "sales"],
}
SINGULAR = {"orders": "order", "invoices": "invoice", "purchases": "purchase", "transactions": "transaction",
            "sales": "sale"}
COUNT_SUFFIXES = [", how many?", " - how many are there?", ", count them", " how many", " count", ", count", " - count",
                  " count", ", how many are there"]
TOTAL_SUFFIXES = [" total", ", total", " sum", " - total", ", what's the total?", " total value", " revenue"]
# nouns for a status, used as "refund count for March" or "pending total"
STATUS_NOUNS = {"refunded": ["refunds", "returns"], "cancelled": ["cancellations", "cancels"], "paid": ["payments"],
                "shipped": ["shipments"], "pending": ["pending ones"], "overdue": ["overdues"]}
SORT_ADJ = {
    "newest": ["latest", "most recent", "newest"],
    "oldest": ["oldest", "earliest"],
    "highest": ["biggest", "largest", "highest value", "most expensive", "priciest", "highest priced"],
    "lowest": ["smallest", "cheapest", "lowest value", "least expensive", "lowest priced"],
}
POSSESSIVE_TIME = {"today": "today's", "yesterday": "yesterday's", "this-week": "this week's",
                   "last-week": "last week's", "this-month": "this month's", "last-month": "last month's"}
CONNECTORS = [" ", " ", ", ", " and ", " "]


SYLLABLES = ["ik", "los", "ri", "pe", "an", "dro", "ni", "kos", "ma", "ev", "ang", "el", "is", "ya", "ze", "phyr", "bright", "water", "kes", "trel", "mon", "tara", "vex", "lum", "or", "ba", "zin", "dra",
             "fel", "quo", "nim", "sol", "ar", "tek", "yu", "suf", "dem", "mir", "cal", "den", "ros", "wick", "har", "lo"]
NAME_SUFFIXES = ["Ltd", "Inc", "Logistics", "Foods", "Systems", "Partners", "Studio", "Group", "Labs", "Co"]


def synthetic_word(rng):
    return "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize()


NUMBER_WORDS = {50: "fifty", 100: "a hundred", 200: "two hundred", 250: "two hundred and fifty", 300: "three hundred",
                400: "four hundred", 500: "five hundred", 600: "six hundred", 750: "seven hundred and fifty",
                800: "eight hundred", 1000: "a thousand", 1500: "fifteen hundred", 2000: "two thousand",
                2500: "twenty five hundred", 3000: "three thousand", 5000: "five thousand", 10000: "ten thousand"}


def money(rng, n):
    """Surface form of an amount. Most forms contain the digits, so the reply can copy the value."""
    r = rng.random()
    if n in NUMBER_WORDS and r < 0.12:
        return rng.choice([NUMBER_WORDS[n], NUMBER_WORDS[n] + " dollars", NUMBER_WORDS[n] + " bucks"])
    r = rng.random()
    if r < 0.4:
        return str(n)
    if r < 0.45 and n >= 1000:
        return f"{n:,}"
    if r < 0.6:
        return f"${n}"
    if r < 0.65 and n >= 1000:
        return f"${n:,}"
    if r < 0.73:
        return f"{n} dollars"
    if r < 0.78:
        return f"{n} bucks"
    if r < 0.83:
        return f"{n} USD"
    if r < 0.9 and n >= 1000 and n % 500 == 0:
        k = n / 1000
        return f"{k:g}k"  # 1.5k, 2k: the reply holds the number, not this word
    return f"${n}"


def customer(rng):
    """Half the names come from a small pool, half are invented, so copying beats remembering."""
    r = rng.random()
    if r < 0.2:
        first = rng.choice(WORLD_NAMES)
        name = f"{first} {rng.choice(LAST)}" if rng.random() < 0.4 else first
    elif r < 0.5:
        first = rng.choice(FIRST)
        name = f"{first} {rng.choice(LAST)}" if rng.random() < 0.5 else first
    else:
        first = synthetic_word(rng)
        r = rng.random()
        if r < 0.4:
            name = first
        elif r < 0.7:
            name = f"{first} {rng.choice(NAME_SUFFIXES)}"
        else:
            name = f"{first} {synthetic_word(rng)}"
    return name.lower() if rng.random() < 0.1 else name


def sample_query(rng, combos):
    """A random structured query: ordered dict of option -> value, plus the subcommand."""
    sub = rng.choices(["find", "count", "total"], weights=[0.6, 0.2, 0.2])[0]
    q = {}
    status, category = rng.choice(combos) if rng.random() < 0.6 else (rng.choice(sorted(STATUSES)), None)
    if rng.random() < 0.7:
        q["status"] = status
    if category is not None and rng.random() < 0.7:
        q["category"] = category
    elif rng.random() < 0.15:
        q["category"] = rng.choice(sorted(CATEGORIES))
    r = rng.random()
    if r < 0.2:
        q["min-amount"] = amount(rng)
    elif r < 0.35:
        q["max-amount"] = amount(rng)
    elif r < 0.45:
        lo, hi = sorted((amount(rng), amount(rng)))
        if lo == hi:
            hi += rng.randint(1, 500)
        q["min-amount"], q["max-amount"] = lo, hi
    r = rng.random()
    if r < 0.2:
        q["when"] = rng.choice(sorted(RELATIVE))
    elif r < 0.3:
        q["last-days"] = small_count(rng)
    elif r < 0.4:
        q["month"] = rng.choice(MONTHS)
        if rng.random() < 0.4:
            q["day"] = rng.randint(1, MONTH_DAYS[MONTHS.index(q["month"])])
        if rng.random() < (0.3 if "day" in q else 0.4):
            q["year"] = rng.choice(YEARS)
    elif r < 0.45:
        q["year"] = rng.choice(YEARS)
    elif r < 0.55:
        which = rng.choice(["before", "after", "before", "after", "both"])
        if which == "both":
            a, b = edge_date(rng), edge_date(rng)
            q["after"], q["before"] = (a, b) if rng.random() < 0.5 else (b, a)
        else:
            q[which] = edge_date(rng)
    if rng.random() < 0.25:
        q["customer"] = customer(rng)
    if sub == "find":
        if rng.random() < 0.3:
            q["sort"] = rng.choice(sorted(SORTS))
        if "sort" in q and rng.random() < 0.3:
            q["limit"] = rng.choice([small_count(rng)] * 3 + [1, 1, 1, 1])
        elif rng.random() < 0.15:
            q["limit"] = small_count(rng)
    if not q and sub == "find":
        q["status"] = status
    return sub, q


ORDER = ["id", "status", "category", "min-amount", "max-amount", "when", "last-days", "month", "day", "year", "after",
         "before", "customer", "sort", "limit"]


def reply(sub, q):
    parts = ["orders", sub]
    for key in ORDER:
        if key in q:
            value = q[key]
            if key == "customer" and " " in str(value):
                value = f'"{value}"'
            parts += [f"--{key}", str(value)]
    return " ".join(parts)


def edge_date(rng):
    """A moment for --before / --after: "october-15", "october-15-2025", "march", "march-2024" or "2024"."""
    r = rng.random()
    if r < 0.15:
        return str(rng.choice(YEARS))
    month = rng.choice(MONTHS)
    value = month
    if r < 0.6:
        value += f"-{rng.randint(1, MONTH_DAYS[MONTHS.index(month)])}"
    if rng.random() < 0.3:
        year = rng.choice(YEARS)
        if value == "february-29" and int(year) % 4:
            value = "february-28"
        value += f"-{year}"
    return value


def edge_text(rng, value):
    """The words for an edge_date value."""
    parts = value.split("-")
    if len(parts) == 1 and parts[0].isdigit():
        return parts[0]
    month = parts[0]
    day = int(parts[1]) if len(parts) > 1 and len(parts[1]) <= 2 else None
    year = parts[-1] if len(parts) > 1 and len(parts[-1]) == 4 else None
    if day:
        return date_text(rng, month, day, year)
    name = month.capitalize() if rng.random() < 0.7 else month
    if rng.random() < 0.2:
        name = MONTH_ABBR[MONTHS.index(month)].capitalize()
    return f"{name} {year}" if year else name


EDGE_WORDS = {
    "before": ["before {t}", "prior to {t}", "earlier than {t}", "placed before {t}", "from before {t}", "before {t}"],
    "after": ["after {t}", "later than {t}", "placed after {t}", "from after {t}", "after {t}"],
}


def join_clauses(rng, clauses):
    """Clauses after "orders": the first follows directly, later ones may be joined with a comma or "and"."""
    text = ""
    for i, clause in enumerate(clauses):
        text += (" " if i == 0 else rng.choice(CONNECTORS)) + clause
    return text


def render_request(rng, sub, q):
    """A request that states every part of the query."""
    pre, post = [], []  # adjectives before "orders", clauses after it
    status_noun = None
    revenue = sub == "total" and rng.random() < 0.2  # "last month's revenue", "revenue from paid orders in March"
    if q.get("status") in STATUS_NOUNS and not revenue and "sort" not in q and "limit" not in q and rng.random() < 0.15:
        status_noun = rng.choice(STATUS_NOUNS[q["status"]])  # "refunds for March" instead of "refunded orders"
    if "status" in q and status_noun:
        pass
    elif "status" in q and revenue:
        post.append("from " + rng.choice(STATUSES[q["status"]][0]) + " orders")
    elif "status" in q:
        adjectives, relatives = STATUSES[q["status"]]
        if rng.random() < 0.55:
            pre.append(rng.choice(adjectives))  # "unpaid orders"
        else:
            post.append(rng.choice(relatives))  # "orders that are still unpaid"
    if "category" in q:
        if rng.random() < 0.3:
            pre.append(category_word(rng, q["category"]))
        else:
            post.append(category_clause(rng, q["category"]))
    lo, hi = q.get("min-amount"), q.get("max-amount")
    if lo is not None and hi is not None:
        post.append(rng.choice(["between {a} and {b}", "from {a} to {b}", "worth {a} to {b}"]).format(a=money(rng, lo), b=money(rng, hi)))
    elif lo is not None:
        post.append(rng.choice(["over {a}", "above {a}", "more than {a}", "of at least {a}", "worth more than {a}", "> {a}", "> {a}", ">= {a}", "{a} or more", "{a}+", "at least {a}", "greater than {a}", "exceeding {a}", "over {a} each"]).format(a=money(rng, lo)))
    elif hi is not None:
        post.append(rng.choice(["under {a}", "below {a}", "less than {a}", "of at most {a}", "worth less than {a}", "< {a}", "< {a}", "<= {a}", "up to {a}", "no more than {a}", "not above {a}", "cheaper than {a}"]).format(a=money(rng, hi)))
    if "when" in q:
        post.append(rng.choice(RELATIVE[q["when"]]))
        if q["when"] in POSSESSIVE_TIME and rng.random() < 0.2:
            post.pop()
            pre.insert(0, POSSESSIVE_TIME[q["when"]])  # "yesterday's cancelled orders"
    elif "last-days" in q:
        post.append(rng.choice(["in the last {n} days", "from the past {n} days", "within the last {n} days",
                                "over the previous {n} days", "during the past {n} days", "since {n} days ago",
                                "that came in over the last {n} days"]).format(n=q["last-days"]))
    elif "day" in q:
        text = date_text(rng, q["month"], q["day"], q.get("year"))
        post.append(rng.choice(["on {t}", "from {t}", "placed on {t}", "{t}", "for {t}", "made on {t}", "on {t}"]).format(t=text))
    elif "month" in q:
        m = q["month"].capitalize() if rng.random() < 0.7 else q["month"]
        if rng.random() < 0.25:
            m = MONTH_ABBR[MONTHS.index(q["month"])].capitalize() + ("." if rng.random() < 0.1 else "")
        text = f"{m} {q['year']}" if "year" in q else m
        post.append(rng.choice(["from {t}", "in {t}", "placed in {t}", "{t}", "{t}", "of {t}", "during {t}"]).format(t=text))
    elif "year" in q:
        post.append(rng.choice(["from {t}", "in {t}", "placed in {t}", "{t}", "{t}", "of {t}", "during {t}"]).format(t=q["year"]))
    for edge in ("after", "before"):
        if edge in q:
            post.append(rng.choice(EDGE_WORDS[edge]).format(t=edge_text(rng, q[edge])))
    possessive = ""
    if "customer" in q:
        if rng.random() < 0.15:
            possessive = f"{q['customer']}'s "  # "Hooli's refunded purchases"
        else:
            post.append(rng.choice(["from {c}", "by {c}", "for customer {c}", "placed by {c}", "for {c}",
                                    "that {c} placed", "{c} bought"]).format(c=q["customer"]))
    rng.shuffle(post)
    noun = status_noun or rng.choice(NOUNS[sub])  # the reply says "orders" whatever the noun
    lead = rng.choice(LEADS[sub])
    sort, limit = q.get("sort"), q.get("limit")
    if revenue and not status_noun:
        noun = rng.choice(["revenue", "revenue", "sales total", "turnover", "income", "earnings", "takings"])
        lead = rng.choice(["give me", "what was", "what's", "show me", "what was our", "", ""])
    if sort and limit and rng.random() < 0.6:
        # "the 5 latest orders", "top 10 biggest invoices", "the largest order" (limit 1)
        prefix = " ".join(pre) + " " if pre else ""
        if limit == 1:
            form = rng.choice(["the {adj} {pre}{one}", "the {adj} {pre}{one}", "just the {adj} {pre}{one}",
                               "the single {adj} {pre}{one}", "{adj} {pre}{one}", "{adj} {pre}{one}"])
            if sort in ("newest", "oldest") and rng.random() < 0.3:
                form = rng.choice(["the last {pre}{one}" if sort == "newest" else "the first {pre}{one}",
                                   "last {pre}{one}" if sort == "newest" else "first {pre}{one}"])
        else:
            form = rng.choice(["the {n} {adj} {pre}{noun}", "top {n} {adj} {pre}{noun}", "{n} {adj} {pre}{noun}",
                               "the {adj} {n} {pre}{noun}", "the {adj} {n} {pre}{noun}"])
            if sort in ("newest", "oldest") and rng.random() < 0.3:
                word = "last" if sort == "newest" else "first"
                form = rng.choice(["{w} {n} {pre}{noun}", "the {w} {n} {pre}{noun}"]).replace("{w}", word)
            elif sort == "highest" and rng.random() < 0.4:
                form = rng.choice(["top {n} {pre}{noun} by amount", "top {n} {pre}{noun} by value", "the top {n} {pre}{noun} by amount"])
            elif sort == "lowest" and rng.random() < 0.4:
                form = rng.choice(["bottom {n} {pre}{noun} by amount", "the {n} {pre}{noun} with the lowest amount",
                                   "the {n} least expensive {pre}{noun}", "the {n} cheapest {pre}{noun}"])
        adj = rng.choice(SORT_ADJ[sort])
        words = form.format(n=limit, adj=adj, noun=noun, one=SINGULAR[noun], pre=prefix)
        text = f"{lead} {possessive}{words}".strip()
        text += join_clauses(rng, post)
    elif sort and not limit and rng.random() < 0.4:
        # "the newest orders", "cheapest unpaid invoices"
        text = f"{lead} {possessive}{rng.choice(SORT_ADJ[sort])} {' '.join(pre + [noun])}".strip()
        text += join_clauses(rng, post)
    else:
        body = " ".join(pre + [noun])
        if sub == "find" and rng.random() < 0.3:
            body = "all " + body
        text = f"{lead} {possessive}{body}".strip()
        text += join_clauses(rng, post)
        if sort:
            text += rng.choice([", ", " "]) + rng.choice(SORTS[sort])
        if limit:
            text += rng.choice([", only the top {n}", ", top {n} only", ", limit {n}", ", first {n}"]).format(n=limit)
    if sub == "count" and (not lead or status_noun and lead not in ("how many", "number of", "the number of")):
        text += rng.choice(COUNT_SUFFIXES)
    elif sub == "total" and not lead:
        text += rng.choice(TOTAL_SUFFIXES)
    text = " ".join(text.split())
    r = rng.random()
    if r < 0.12 and text:
        text = rng.choice(["hey, ", "hi, ", "ok so ", "so ", "quick one: ", "um, ", "right, ", "listen, "]) + text
    elif r < 0.24 and text:
        text += rng.choice([" please", " thanks", " if you can", " for me", " when you get a chance", " asap", " please?"])
    r = rng.random()
    if r < 0.15:
        text += "?" if sub == "count" else "."
    elif r < 0.2:
        text = text[0].upper() + text[1:]
    return text


ID_FORMS = ["confirmation number {id}", "reference {id}", "order ref {id}", "ticket {id}", "can you pull up order {id}",
            "I need the details of order {id}", "what's in order {id}", "order {id} please", "check order {id}", "status of order {id}",
            "order {id}", "order #{id}", "order number {id}", "order no. {id}", "#{id}", "show order {id}",
            "find order {id}", "look up order {id}", "get order {id}", "show me order {id}", "details of order {id}",
            "invoice {id}", "invoice #{id}", "open order {id}", "what is order {id}", "order id {id}"]


def order_id(rng):
    return rng.choice([rng.randint(1, 999), rng.randint(1000, 99999), rng.randint(10000, 999999)])


def render_id(rng, q):
    text = rng.choice(ID_FORMS).format(id=q["id"])  # never a bare number: it could be anything
    return text + rng.choice(["?", "."]) if rng.random() < 0.15 else text


def terse_amount(rng, key, n):
    if key == "min-amount":
        form = rng.choice(["over {a}", "> {a}", "above {a}", "more than {a}", "{a}+", "at least {a}", ">= {a}", "over {a}", "greater than {a}"])
    else:
        form = rng.choice(["under {a}", "< {a}", "below {a}", "less than {a}", "up to {a}", "at most {a}", "<= {a}", "under {a}", "cheaper than {a}"])
    return form.format(a=money(rng, n))


def render_terse(rng, q):
    """A search-box request for a find: keywords with no sentence around them ("unpaid", "over 500", "Oct 15",
    "customer Globex", "refunded toys march"). Sometimes "orders" comes first."""
    parts = []
    if "status" in q:
        parts.append(rng.choice(STATUSES[q["status"]][0]))
    if "category" in q:
        parts.append(category_word(rng, q["category"], plain=0.7))
    for key in ("min-amount", "max-amount"):
        if key in q:
            parts.append(terse_amount(rng, key, q[key]))
    if "when" in q:
        parts.append(RELATIVE[q["when"]][0])
    elif "last-days" in q:
        parts.append(rng.choice(["last {n} days", "past {n} days", "{n} days", "last {n} days"]).format(n=q["last-days"]))
    elif "day" in q:
        parts.append(date_text(rng, q["month"], q["day"], q.get("year")))
    elif "month" in q:
        m = q["month"].capitalize() if rng.random() < 0.6 else q["month"]
        if rng.random() < 0.3:
            m = MONTH_ABBR[MONTHS.index(q["month"])].capitalize()
            if m == "Sep" and rng.random() < 0.5:
                m = "Sept"
        parts.append(f"{m} {q['year']}" if "year" in q else m)
    elif "year" in q:
        parts.append(q["year"])
    for edge in ("after", "before"):
        if edge in q:
            parts.append(rng.choice([edge, edge, {"before": "prior to", "after": "later than"}[edge]]) + " " + edge_text(rng, q[edge]))
    if "customer" in q:
        c = q["customer"]
        parts.append(rng.choice([f"customer {c}", f"customer {c}", c, f"from {c}", f"by {c}", f"{c}'s orders"]) if len(parts) == 0 else
                     rng.choice([f"customer {c}", c, f"from {c}", f"by {c}", f"{c}'s"]))
    if "sort" in q:
        parts.append(rng.choice(SORT_ADJ[q["sort"]]))
    if "limit" in q:
        parts.append(rng.choice(["top {n}", "first {n}", "limit {n}"]).format(n=q["limit"]))
    rng.shuffle(parts)
    text = " ".join(parts)
    r = rng.random()
    if parts and r < 0.25:
        text = rng.choice(["orders", "orders", "all", "show"]) + " " + text
    return text.strip() or rng.choice(["all", "all orders", "orders", "everything", "show all", "show all orders",
                                       "list all orders", "all of them", "show everything", "every order"])


TERSE_UNITS = {
    "status": ["status"], "category": ["category"], "amount": ["min-amount", "max-amount"],
    "date": ["when", "last-days", "month", "day", "year", "after", "before"], "customer": ["customer"],
    "sort": ["sort", "limit"],
}


def chatty(rng, combos):
    """Short conversational requests that name one or two parts without "orders": (text, query)."""
    r = rng.random()
    status = rng.choice(sorted(STATUSES))
    adj = rng.choice(STATUSES[status][0])
    c = customer(rng)
    if rng.random() < 0.2:  # "last 3 orders", "first order", "earliest order"
        newest = rng.random() < 0.6
        n = rng.choice([1, 1, 2, 3, 3, 4, 5, 5, 6, 8, 10, 12, 15, 20])
        noun = rng.choice(["orders", "orders", "orders", "invoices", "purchases"])
        one = SINGULAR[noun]
        if n == 1:
            word = rng.choice(["last", "latest", "newest", "most recent"] if newest else ["first", "earliest", "oldest"])
            text = rng.choice(["{w} {one}", "the {w} {one}", "{w} {one}", "show the {w} {one}"]).format(w=word, one=one)
        else:
            word = rng.choice(["last", "latest", "newest", "most recent"] if newest else ["first", "earliest", "oldest"])
            text = rng.choice(["{w} {n} {noun}", "the {w} {n} {noun}", "show the {w} {n} {noun}", "{w} {n} {noun}"]).format(w=word, n=n, noun=noun)
        return text, {"sort": "newest" if newest else "oldest", "limit": n}
    if r < 0.2:
        cat = rng.choice(sorted(CATEGORIES))
        return rng.choice(["only {x}", "just {x}", "{x} only", "show only {x}", "only the {x} ones"]).format(x=cat), {"category": cat}
    if r < 0.4:
        text = rng.choice(["what has {c} ordered", "what did {c} order", "anything from {c}", "everything {c} bought",
                           "what has {c} bought", "what did {c} buy", "has {c} ordered anything", "{c}'s orders"]).format(c=c)
        return text, {"customer": c}
    if r < 0.6:
        text = rng.choice(["what's {a}", "what is {a}", "anything {a}", "is anything {a}", "what's still {a}", "is there anything {a}"]).format(a=adj)
        return text, {"status": status}
    if r < 0.8:
        text = rng.choice(["is anything {a} from {c}", "anything {a} from {c}", "what's {a} for {c}", "does {c} have anything {a}"]).format(a=adj, c=c)
        return text, {"status": status, "customer": c}
    when = rng.choice(sorted(RELATIVE))
    text = rng.choice(["what did {c} order {w}", "what has {c} bought {w}", "everything {c} ordered {w}"]).format(c=c, w=RELATIVE[when][0])
    return text, {"customer": c, "when": when}


def terse_query(rng, combos):
    """A find query cut down to one to three of its parts; sometimes empty ("all")."""
    r = rng.random()
    if r < 0.05:
        return {}
    if r < 0.17:  # a calendar day alone: "oct 15", "sept 20"
        month = rng.choice(MONTHS)
        q = {"month": month, "day": rng.randint(1, MONTH_DAYS[MONTHS.index(month)])}
        if rng.random() < 0.2:
            q["year"] = rng.choice(YEARS)
        return q
    if r < 0.25:  # a status or a name alone
        return {"status": rng.choice(sorted(STATUSES))} if rng.random() < 0.5 else {"customer": customer(rng)}
    while True:
        _, q = sample_query(rng, combos)
        units = [u for u, keys in TERSE_UNITS.items() if any(k in q for k in keys)]
        if not units:
            continue
        keep = rng.sample(units, min(len(units), rng.choice([1, 1, 1, 2, 2, 3])))
        out = {k: q[k] for u in keep for k in TERSE_UNITS[u] if k in q}
        if "limit" in out and "sort" not in out:
            del out["limit"]
        if "limit" in out and rng.random() < 0.3:
            del out["limit"]
        return out


UNKNOWN = "orders unknown"  # the reply to anything that is not an order search
CITIES = ["Paris", "Tokyo", "Lagos", "Lima", "Oslo", "Cairo", "Berlin", "Mumbai", "Sydney", "Toronto", "Athens", "Seoul"]
TOPICS = ["the sea", "autumn", "coffee", "a robot", "my cat", "space travel", "friendship", "the moon", "trains", "rain"]
FOODS = ["pasta", "pancakes", "bread", "curry", "soup", "a salad", "rice", "cookies"]
LANGS = ["French", "Spanish", "German", "Greek", "Italian", "Japanese"]
SMALL_TALK = ["hello", "hi", "hi there", "hey", "good morning", "good evening", "thanks", "thank you", "ok", "yes", "no",
              "how are you", "how are you doing today", "what can you do", "who are you", "who made you", "what is your name",
              "tell me a joke", "tell me something funny", "help", "help me", "are you a robot", "goodbye", "see you later",
              "what's up", "can you help me", "test", "testing 123", "lol", "never mind"]
OOS_TEMPLATES = [
    "what's the weather in {city}", "what is the weather like in {city} today", "will it rain in {city} tomorrow",
    "what time is it in {city}", "how far is {city} from here", "what is the capital of {city}",
    "how many people live in {city}", "what is {n} plus {m}", "what's {n} times {m}", "calculate {n} divided by {m}",
    "write a poem about {topic}", "write a short story about {topic}", "tell me a fact about {topic}",
    "how do I make {food}", "give me a recipe for {food}", "what is a good side dish for {food}",
    "translate {food} to {lang}", "how do you say thank you in {lang}", "translate hello into {lang}",
    "recommend a good movie", "recommend a book about {topic}", "suggest a name for my dog",
    "who won the football match yesterday", "what's the best programming language", "explain quantum physics",
    "how do I reset my password", "how do I change my email address", "what is the meaning of life",
    "delete order {id}", "delete all my orders", "cancel order {id}", "please cancel my order number {id}",
    "refund order {id}", "refund the customer {name}", "email {name} about their order", "send {name} an invoice",
    "update the address for {name}", "change the shipping address of order {id}", "create a new order for {name}",
    "place an order for {n} {food}", "mark order {id} as shipped", "print invoice {id}", "export all orders to a spreadsheet",
    "add a note to order {id}", "approve order {id}", "archive order {id}", "reopen order {id}", "edit order {id}",
    "duplicate order {id}", "reorder order {id}", "pay for order {id}", "void invoice {id}", "resend invoice {id}",
    "hold order {id}", "escalate order {id}", "assign order {id} to {name}", "split order {id} in two", "remove {name} from the customer list", "what is the status of my delivery",
    "when will my package arrive", "track order {id}", "where is my parcel", "what is your refund policy",
    "do you ship to {city}", "how do I contact support", "what are your opening hours", "can I pay by card",
    "ship order {id}", "ship order {id} now", "ship {name}'s order", "send order {id}", "dispatch order {id}",
    "send out order {id} today", "email the customer about order {id}", "email {name} about order {id}",
    "notify {name} about order {id}", "call {name} about order {id}", "message the customer of order {id}",
    "text {name}", "call {name}", "process order {id}", "fulfill order {id}", "pack order {id}", "close order {id}",
    "flag order {id}", "mark order {id} as paid", "set order {id} to cancelled", "change the status of order {id}",
    "update order {id}", "apply a discount to order {id}", "charge order {id} again", "retry the payment for order {id}",
    "play some music", "play {topic}", "play a song about {topic}", "open the settings", "open my email",
    "turn on the lights", "turn up the volume", "set a timer for {n} minutes", "show me pictures of {topic}",
    "show me a video about {topic}", "find me a good {food} place", "find a restaurant near me", "find a hotel in {city}",
    "list the planets", "list some {food} ideas", "list the days of the week", "get me a coffee", "get me directions to {city}",
    "give me a {food} recipe", "search for {topic}", "look up the weather", "look up {city} on the map", "fetch the news",
    "pull up my calendar", "display the time", "let me see a cat", "any good jokes", "what should I eat for dinner",
    "how many days until christmas", "count to {n}", "how many people live in {city}", "total of {n} and {m}",
    "sum up this paragraph for me", "what is the total population of {city}", "how many calories in {food}",
]
# Wordings the model never sees in training, to test that "not an order search" generalizes.
OOS_HELDOUT = [
    "what's the temperature outside", "sing me a song", "what year did the war end", "how tall is mount everest",
    "can you book me a flight to {city}", "turn off the lights", "I'm feeling sad", "open the settings page",
    "remind me to call {name} tomorrow", "set an alarm for seven", "who is the president", "what is a good name for a bakery",
    "close my account", "get order {id} out of the door today", "write an email to {name}", "give me a joke about {topic}",
    "what does {food} taste like", "asdf qwerty", "zxcv", "????",
]


KEYBOARD = ["asdf", "qwerty", "zxcv", "hjkl", "uiop", "lkjh", "mnbv"]


def gibberish(rng):
    """Text that means nothing: keyboard runs, random letters, punctuation or digits only."""
    r = rng.random()
    if r < 0.3:
        return " ".join(rng.choice(KEYBOARD) + "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=rng.randint(0, 4)))
                        for _ in range(rng.randint(1, 3)))
    if r < 0.55:
        return " ".join("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=rng.randint(2, 9))) for _ in range(rng.randint(1, 3)))
    if r < 0.8:
        return "".join(rng.choices("?!.,;:-_*#@", k=rng.randint(1, 6)))
    return " ".join(str(rng.randint(0, 99999)) for _ in range(rng.randint(1, 3)))


SEARCH_WORDS = ({w for adj, _ in STATUSES.values() for a in adj for w in a.split()} | set(CATEGORIES) | set(MONTHS)
                | set(MONTH_ABBR) | {"all", "order", "orders", "customer", "invoice", "invoices", "today", "yesterday",
                                      "week", "month", "year", "before", "after", "newest", "oldest", "cheapest"})


def split_off_topic(messages):
    """(train, heldout) halves of generated off-topic messages; a message always lands on the same side.
    Short fragments made of search words ("unpaid", "march") are dropped: they are searches now."""
    messages = [m for m in messages
                if not (len(m.split()) <= 4 and any(w.strip(".,!?'\"") in SEARCH_WORDS for w in m.lower().split()))]
    train = [m for m in messages if stable_hash(m, "off-topic") % 10 != 0]
    held = [m for m in messages if stable_hash(m, "off-topic") % 10 == 0]
    return train, held


def out_of_scope(rng, heldout=False, pool=()):
    """A request that is not an order search; the reply is "orders unknown". pool holds generated
    off-topic messages (the train or the held-out side of split_off_topic)."""
    r = rng.random()
    if r < 0.1:
        return gibberish(rng)
    if pool and r < 0.7:
        return rng.choice(pool)
    if not heldout and r < 0.8:
        return rng.choice(SMALL_TALK)
    text = rng.choice(OOS_HELDOUT if heldout else OOS_TEMPLATES)
    name = customer(rng)
    return text.format(city=rng.choice(CITIES), n=rng.randint(2, 99), m=rng.randint(2, 20), topic=rng.choice(TOPICS),
                       food=rng.choice(FOODS), lang=rng.choice(LANGS), id=rng.randint(100, 99999), name=name)


KEYBOARD_NEIGHBOURS = {"a": "qsz", "e": "wrd", "i": "uok", "o": "ipl", "u": "yij", "s": "adw", "t": "ryg", "n": "bm",
                       "r": "etf", "d": "sfe", "l": "ko", "m": "nk", "p": "o", "c": "xv", "h": "gj", "g": "fh", "y": "tu"}


def add_noise(rng, text, protect):
    """Typing mistakes in the request: dropped apostrophes and a typo in one or two longer words. Words
    in protect (customer names) and anything with a digit are left alone so the reply can still copy them."""
    if "'" in text and rng.random() < 0.5:
        text = " ".join(w if any(p in w.lower() for p in protect) else w.replace("'", "") for w in text.split(" "))
    words = text.split(" ")
    number_words = {w for phrase in NUMBER_WORDS.values() for w in phrase.split()}  # a spelled-out amount must stay readable
    eligible = [i for i, w in enumerate(words)
                if len(w) >= 4 and w.isalpha() and w.lower() not in protect and w.lower() not in number_words
                and not any(p in w.lower() for p in protect)]
    for i in rng.sample(eligible, min(len(eligible), rng.choice([1, 1, 2]))):
        w, k = words[i], rng.randrange(1, len(words[i]) - 1)
        op = rng.choice(["drop", "double", "swap", "near"])
        if op == "drop":
            w = w[:k] + w[k + 1:]
        elif op == "double":
            w = w[:k] + w[k] + w[k:]
        elif op == "swap":
            w = w[:k] + w[k + 1] + w[k] + w[k + 2:]
        elif w[k].lower() in KEYBOARD_NEIGHBOURS:
            w = w[:k] + rng.choice(KEYBOARD_NEIGHBOURS[w[k].lower()]) + w[k + 1:]
        words[i] = w
    return " ".join(words)


def heldout_combos(rng, fraction):
    """(status, category) pairs that never appear in training."""
    pairs = [(s, c) for s in sorted(STATUSES) for c in sorted(CATEGORIES)]
    return {p for p in pairs if stable_hash(f"{p[0]}/{p[1]}", "combo") % 1000 < fraction * 1000}


def build(n_train, n_heldout, seed, heldout_fraction=0.1, oos_fraction=0.07, off_topic=(), noise=0.15):
    rng = seeded_rng(seed)
    off_train, off_held = split_off_topic(off_topic)
    held = heldout_combos(rng, heldout_fraction)
    train_combos = [(s, c) for s in sorted(STATUSES) for c in sorted(CATEGORIES) if (s, c) not in held]
    held_combos = sorted(held)
    train, seen = [], {}

    def one(combos):
        r = rng.random()
        lower = rng.random() < 0.12  # typed in lower case, names included
        sub = "find"
        chat = None
        if r < 0.05:
            q = {"id": order_id(rng)}
        elif r >= 0.17 and r < 0.23 and combos is train_combos:
            chat = chatty(rng, combos)
            q = chat[1]
        elif r < 0.17:
            q = terse_query(rng, combos)
        else:
            sub, q = sample_query(rng, combos)
        if lower and "customer" in q:
            q["customer"] = q["customer"].lower()
        if chat:
            text = chat[0]
        elif "id" in q:
            text = render_id(rng, q)
        elif r < 0.17:
            text = render_terse(rng, q)
        else:
            text = render_request(rng, sub, q)
        if lower:
            text = text.lower()
        if noise and rng.random() < noise:
            protect = {w.strip('"').lower() for w in str(q.get("customer", "")).split()}
            text = add_noise(rng, text, protect)
        return text, reply(sub, q), q

    while len(train) < n_train:
        if rng.random() < oos_fraction:
            train.append({"instruction": out_of_scope(rng, pool=off_train), "output": UNKNOWN})
            continue
        inst, out, q = one(train_combos)
        if (q.get("status"), q.get("category")) in held:
            continue
        train.append({"instruction": inst, "output": out})
    train_keys = {(p["instruction"], p["output"]) for p in train}
    heldout = []
    while len(heldout) < n_heldout:
        if rng.random() < oos_fraction * 2:
            heldout.append({"instruction": out_of_scope(rng, heldout=True, pool=off_held), "output": UNKNOWN, "split": "out_of_scope"})
            continue
        unseen = rng.random() < 0.5
        inst, out, q = one(held_combos if unseen else train_combos)
        if unseen and (q.get("status"), q.get("category")) not in held:
            continue
        if not unseen and (q.get("status"), q.get("category")) in held:
            continue
        if (inst, out) in train_keys:
            continue
        heldout.append({"instruction": inst, "output": out, "split": "unseen_combo" if unseen else "seen_combo"})
    return train, heldout


def option_probe(command):
    """(prefix, " option") for the first option after the subcommand."""
    parts = command.split()
    for i, tok in enumerate(parts[2:], start=2):
        if tok.startswith("--"):
            return " ".join(parts[:i]), " " + tok
        break
    return None


def build_tasks(heldout, per_category, rng):
    tasks = collections.defaultdict(list)
    rng.shuffle(heldout)
    for ex in heldout:
        inst, out = ex["instruction"], ex["output"]
        if ex["split"] == "out_of_scope":
            tasks["Query not an order search"].append(probes.probe("", [out], inst, complete=True))
            continue
        label = "unseen combination" if ex["split"] == "unseen_combo" else "seen combination"
        tasks[f"Query full reply ({label})"].append(probes.probe("", [out], inst, complete=True))
        parts = out.split()
        tasks[f"Query subcommand ({label})"].append(probes.probe("orders", [" " + parts[1]], inst))
        opt = option_probe(out)
        if opt:
            tasks[f"Query first option ({label})"].append(probes.probe(opt[0], [opt[1]], inst))
        cp = probes.copy_probe(inst, out)
        if cp:
            tasks[f"Query argument copy ({label})"].append(probes.probe(cp[0], [cp[1]], inst))
    result = []
    for category, items in sorted(tasks.items()):
        unique = list({json.dumps(t, sort_keys=True): t for t in items}.values())[:per_category]
        result.extend({"category": category, **t} for t in unique)
        print(f"  {category}: {len(unique)}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train", type=int, default=60000)
    ap.add_argument("--heldout", type=int, default=3000)
    ap.add_argument("--per-category", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--off-topic", help="JSON list of off-topic requests from generate_off_topic.py")
    ap.add_argument("--oos-fraction", type=float, default=0.07, help="share of training pairs that are off-topic")
    ap.add_argument("--noise", type=float, default=0.15, help="share of requests with typing mistakes (0 = none)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    off_topic = json.load(open(args.off_topic)) if args.off_topic else []
    train, heldout = build(args.train, args.heldout, args.seed, oos_fraction=args.oos_fraction, off_topic=off_topic,
                           noise=args.noise)
    save_json(os.path.join(args.out_dir, "query_sft.json"), train)
    save_json(os.path.join(args.out_dir, "query_heldout.json"), heldout)
    print(f"{len(train)} train pairs, {len(heldout)} held-out pairs")
    tasks = build_tasks(heldout, args.per_category, seeded_rng(args.seed + 1))
    save_json(os.path.join(args.out_dir, "query_tasks.json"), tasks)
    print(f"{len(tasks)} probes")


if __name__ == "__main__":
    main()
