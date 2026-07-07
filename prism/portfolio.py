"""Portfolio construction: PRISM Full / Select / Edge / Custom."""


def build_portfolios(df, cfg):
    portfolios = {}
    n = len(df)
    nf = max(int(n * 0.50), min(8, n))
    ds = df.sort_values("prism_score", ascending=False)

    pf = ds.head(nf).copy()
    portfolios["PRISM Full"] = pf["Stock"].tolist()

    frag = pf[pf["fragile_flag"]]["Stock"].tolist()
    base = pf[~pf["fragile_flag"]]["Stock"].tolist()
    nr = len(frag)
    cands = ds[(~ds["Stock"].isin(portfolios["PRISM Full"]))
               & (~ds["fragile_flag"])].head(max(nr + 3, 5)).copy()
    if len(cands) > 0:
        cands["rv"] = (cands["prism_score"] * 0.5 + cands["stretch_score"] * 0.3
                       + (100 - cands["Beta"].fillna(1.5) * 40) * 0.2)
        reps = cands.nlargest(nr, "rv")["Stock"].tolist()
    else:
        reps = []
    portfolios["PRISM Select"] = base + reps

    es = min(cfg["edge_size"], len(portfolios["PRISM Select"]))
    sdf = df[df["Stock"].isin(portfolios["PRISM Select"])].copy()
    ms = sdf["prism_score"].median() * 0.85
    ee = sdf[(sdf["prism_score"] >= ms) & (sdf["Average Outcome"] >= 0)].copy()
    if len(ee) < es:
        ee = sdf[sdf["Average Outcome"] >= 0].copy()
    if len(ee) > 0:
        def norm(s):
            r = s.max() - s.min()
            return (s - s.min()) / r if r > 0 else 0.5
        eq = ee["Average Outcome"] * (ee["stretch_score"] / 100)
        ee["edge_score"] = (norm(ee["prism_score"]) * 0.40 + norm(eq) * 0.25
                            + norm(ee["blended_growth"]) * 0.25
                            + norm(1 - ee["52w_position"]) * 0.10)
        portfolios["PRISM Edge"] = ee.nlargest(es, "edge_score")["Stock"].tolist()
    else:
        portfolios["PRISM Edge"] = sdf.nlargest(es, "prism_score")["Stock"].tolist()

    custom = [s for s in portfolios["PRISM Edge"] if s not in cfg["force_exclude"]]
    for s in cfg["force_include"]:
        if s not in custom and s in df["Stock"].values:
            custom.append(s)
    portfolios["Custom"] = custom
    return portfolios, frag, reps
