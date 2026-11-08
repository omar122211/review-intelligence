"""YelpCamp Review Intelligence dashboard.  Run the API first, then:  streamlit run ui/streamlit_app.py"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
MOOD = {"positive": "🟢 Positive", "neutral": "🟡 Neutral", "negative": "🔴 Negative"}
EXAMPLES = ["is it quiet at night?", "clean bathrooms", "worth the money?", "good for kids"]

st.set_page_config(page_title="YelpCamp · Review Intelligence", page_icon="🏕️", layout="wide")


def api(method: str, path: str, **kw):
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=90, **kw)
    except httpx.HTTPError:
        return None, "The review service is not reachable. Is the API running?"
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = None
        return None, detail if isinstance(detail, str) else "Something went wrong — please try again."
    return r.json(), None


def stars(n: int) -> str:
    return "★" * n + "☆" * (5 - n)


# ---------------------------------------------------------------- sidebar
camps, err = api("GET", "/campgrounds")
health, _ = api("GET", "/health")
with st.sidebar:
    st.header("🏕️ YelpCamp")
    st.caption("Review Intelligence for campground owners")
    if err:
        st.error(err)
        st.stop()
    names = {c["id"]: f"{c['name']} · {c['average_stars']}★" for c in camps}
    cg_id = st.radio("Campground", list(names), format_func=names.get)
    st.divider()
    if health:
        st.caption(f"Sentiment model: **{health['sentiment_model']}** (runs locally)")
    st.caption("ℹ️ The campgrounds and reviews on this page are **AI-generated demo data** — YelpCamp has no real "
               "reviews yet. All accuracy numbers come from real human reviews (see the project README).")
    st.caption("🔒 Text you analyse is sent to the free Gemini API only when you press an AI button.")

cg, err = api("GET", f"/campgrounds/{cg_id}")
if err:
    st.error(err)
    st.stop()
reviews = {r["id"]: r for r in cg["reviews"]}

st.title(cg["name"])
avg = sum(r["stars"] for r in cg["reviews"]) / max(1, len(cg["reviews"]))
st.caption(f"📍 {cg['location']} · {stars(round(avg))} {avg:.1f} from {len(cg['reviews'])} reviews")

tab_insights, tab_search, tab_analyze, tab_reviews = st.tabs(["Insights", "Search reviews", "Analyse a review", "All reviews"])

# ---------------------------------------------------------------- insights
with tab_insights:
    s = cg.get("summary")
    if s:
        st.subheader(s["headline"])
        left, right = st.columns(2)
        for col, title, points in ((left, "👍 Campers like", s["pros"]), (right, "👎 Campers complain about", s["cons"])):
            with col:
                st.markdown(f"**{title}**")
                for p in points:
                    with st.expander(f"{p['text']} — {len(p['reviews'])} review{'s' if len(p['reviews']) != 1 else ''}"):
                        for i in p["reviews"]:
                            r = reviews.get(i)
                            if r:
                                st.markdown(f"{stars(r['stars'])} **{r['author']}** — {r['text']}")
        st.caption(f"Best for: {s['best_for']}")
    else:
        st.info("No AI summary yet for this campground.")

    st.markdown("#### What campers say about each aspect")
    st.caption("From the aspects each review actually mentions (AI-read, every mention backed by a quote).")
    for a in cg["aspects"]:
        c1, c2 = st.columns([1.2, 3])
        c1.markdown(f"**{a['label']}**  \n<small>{a['mentions']} mentions</small>", unsafe_allow_html=False)
        if a["mentions"]:
            c2.progress(a["positive_share"], text=f"{a['positive']} 👍 · {a['negative']} 👎 · {a['neutral']} 😐")
        else:
            c2.caption("not mentioned")

    pred = [r["predicted"]["sentiment"] for r in cg["reviews"]]
    agree = sum(p == ("negative" if r["stars"] <= 2 else "neutral" if r["stars"] == 3 else "positive")
                for p, r in zip(pred, cg["reviews"]))
    m1, m2, m3 = st.columns(3)
    m1.metric("Positive reviews", pred.count("positive"))
    m2.metric("Negative reviews", pred.count("negative"))
    m3.metric("Model agrees with the stars", f"{agree}/{len(pred)}")

# ---------------------------------------------------------------- search
with tab_search:
    st.caption("Searches by meaning, not exact words — “quiet at night” also finds “we slept like babies”.")
    cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if cols[i].button(ex, key=f"ex{i}", width="stretch"):
            st.session_state.query = ex
    q = st.text_input("Ask the reviews", key="query", max_chars=200, placeholder="e.g. are the showers hot?")
    if q and len(q.strip()) >= 2:
        hits, err = api("GET", f"/campgrounds/{cg_id}/search", params={"q": q, "k": 5})
        if err:
            st.error(err)
        for h in hits or []:
            with st.container(border=True):
                st.markdown(f"{stars(h['stars'])} **{h['author']}** · match {h['score']:.2f}")
                st.write(h["text"])

# ---------------------------------------------------------------- analyse
with tab_analyze:
    text = st.text_area("Paste a review", height=140, max_chars=3000,
                        placeholder="The view was amazing but the toilets were filthy and the music went on until 2am…")
    b1, b2, b3 = st.columns(3)
    run_local = b1.button("Analyse (local model)", type="primary", width="stretch")
    run_explain = b2.button("✨ Explain with AI", width="stretch")
    run_reply = b3.button("✉️ Draft owner reply", width="stretch")
    if text.strip() and (run_local or run_explain):
        res, err = api("POST", "/analyze", params={"explain": str(run_explain).lower()}, json={"text": text})
        if err:
            st.error(err)
        else:
            sres = res["local"]["sentiment"]
            st.markdown(f"### {MOOD[sres['sentiment']]} · {sres['confidence']:.0%} sure")
            st.caption(f"{res['local']['model']} · {res['latency_ms']} ms")
            if res.get("explanation"):
                st.markdown("**What the review says, with quotes**")
                for m in res["explanation"]["aspects"]:
                    st.markdown(f"- {MOOD[m['sentiment']]} **{m['aspect'].replace('_', ' & ')}** — “{m['quote']}”")
                if not res["explanation"]["aspects"]:
                    st.caption("No specific aspects mentioned.")
            elif res.get("explanation_error"):
                st.info(res["explanation_error"])
            with st.expander("Aspect estimates from the local model"):
                st.caption("Predicted for every aspect, even ones the review doesn't mention.")
                st.table({"Aspect": [a["label"] for a in res["local"]["aspects"]],
                          "Estimate": [f"{MOOD[a['sentiment']]} ({a['confidence']:.0%})" for a in res["local"]["aspects"]]})
    if text.strip() and run_reply:
        res, err = api("POST", "/reply", json={"text": text, "campground": cg["name"]})
        if err:
            st.error(err)
        else:
            st.markdown("**Suggested reply** — always read and edit before posting")
            st.text_area("Reply", res["reply"], height=180, label_visibility="collapsed")

# ---------------------------------------------------------------- all reviews
with tab_reviews:
    for r in cg["reviews"]:
        exp = cg["explanations"].get(str(r["id"]), {})
        chips = " ".join(f"`{m['aspect'].replace('_', ' & ')} {'+' if m['sentiment'] == 'positive' else '−' if m['sentiment'] == 'negative' else '~'}`"
                         for m in exp.get("aspects", []))
        with st.container(border=True):
            st.markdown(f"{stars(r['stars'])} **{r['author']}** · model: {MOOD[r['predicted']['sentiment']]}  {chips}")
            st.write(r["text"])
