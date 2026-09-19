"""Rewrite generated where-clauses into the gold FELN.json surface style.

Gold style (verified by `python normalize.py FELN.json`, which asserts every
parseable gold clause is a fixed point):

    subtype = cast(..) and (a) and (b)      subtype atom leads, bare; others wrapped
    (a) and (b)                              no subtype: every AND operand wrapped
    col = 'x' or col = 'y'                   same-column OR is one bare value set
    (a) or (b)                               different-column OR: operands wrapped
    LIKE -> ILIKE                            on the 11 columns the OKF hints name

Clauses with BETWEEN are rare in both sets and are left to the caller to drop.
"""
import re
import sys

ILIKE_COLS = {"discovery_name", "discovery_wellbore_name", "drilling_facility", "drilling_operator",
              "included_in_discovery_name", "main_grouping", "medium", "pipe_name", "production_licence",
              "well_name", "wellbore_name"}
SUBTYPE = re.compile(r"^(?:discovery_type|content_type|PipelinesType) = cast\(")
ATOM = re.compile(r"(\w+) (=|<>|>=|<=|>|<|ILIKE|LIKE) (cast\([^)]*\)|timestamp '[^']*'|'(?:[^']|'')*'|-?[\d.]+)")
TOK = re.compile(r"\(|\)|\band\b|\bor\b|\bAND\b|" + ATOM.pattern)  # uppercase AND = date-range pair, kept verbatim


class Unparseable(ValueError):
    pass


def tokens(s):
    pos = 0
    for m in TOK.finditer(s):
        if s[pos:m.start()].strip():
            raise Unparseable(s)
        pos = m.end()
        yield m
    if s[pos:].strip():
        raise Unparseable(s)


def parse(s):
    """-> ('atom', col, op, val) | ('and'|'or', [nodes])"""
    toks = list(tokens(s))
    i = 0

    def expr():
        nonlocal i
        items, ops = [term()], []
        while i < len(toks) and toks[i].group(0) in ("and", "or", "AND"):
            ops.append(toks[i].group(0)); i += 1
            items.append(term())
        if not ops:
            return items[0]
        if len(set(ops)) != 1:
            raise Unparseable(s)  # mixed and/or without parens
        return (ops[0], items)

    def term():
        nonlocal i
        t = toks[i]
        if t.group(0) == "(":
            i += 1
            node = expr()
            if toks[i].group(0) != ")":
                raise Unparseable(s)
            i += 1
            return node
        if t.group(1) is None:
            raise Unparseable(s)
        i += 1
        return ("atom", t.group(1), t.group(2).upper(), t.group(3))

    node = expr()
    if i != len(toks):
        raise Unparseable(s)
    return node


def flatten(node):
    if node[0] in ("and", "or"):
        out = []
        for c in node[1]:
            c = flatten(c)
            out += c[1] if c[0] == node[0] and not is_valueset(c) else [c]
        return (node[0], out)
    return node


def atom_str(a):
    _, col, op, val = a
    if op == "LIKE" and col in ILIKE_COLS:
        op = "ILIKE"
    return f"{col} {op} {val}"


def is_valueset(node):
    return node[0] == "or" and all(c[0] == "atom" and c[2] == "=" for c in node[1]) and len({c[1] for c in node[1]}) == 1


def bare(node):
    """render without outer parens: atom, value-set, or inner and/or"""
    if node[0] == "atom":
        return atom_str(node)
    if is_valueset(node):
        return " or ".join(atom_str(c) for c in node[1])
    if node[0] == "AND":
        return " AND ".join(bare(c) for c in node[1])
    return render(node)


def render(node):
    node = flatten(node)
    if node[0] == "atom" or is_valueset(node):
        return bare(node)
    kids = node[1]
    if node[0] == "and":
        kids = sorted(kids, key=lambda k: 0 if (k[0] == "atom" and SUBTYPE.match(atom_str(k))) else 1)
        parts = []
        for j, k in enumerate(kids):
            s = bare(k)
            parts.append(s if (j == 0 and k[0] == "atom" and SUBTYPE.match(s)) else f"({s})")
        return " and ".join(parts)
    return " or ".join(f"({bare(k)})" for k in kids)


def normalize(where):
    return "" if where == "" else render(parse(where))


if __name__ == "__main__":
    import json
    recs = json.load(open(sys.argv[1], encoding="utf-8"))
    same = diff = skip = 0
    for r in recs:
        for w in r["meta"]["where"]:
            if "BETWEEN" in w:
                skip += 1; continue
            try:
                n = normalize(w)
            except Unparseable:
                skip += 1; print("UNPARSEABLE", w); continue
            if n == w:
                same += 1
            else:
                diff += 1; print("DIFF\n  ", w, "\n  ", n)
    print(f"fixed point: {same}, changed: {diff}, skipped: {skip}")
    assert diff <= 0.01 * (same + diff), "gold is not a fixed point of normalize()"
