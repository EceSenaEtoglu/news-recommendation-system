import os
import asyncio
import streamlit as st
import subprocess, sys, json, pathlib
import uuid

# Assume src.data_models and other local modules are available in the path
# If running this script standalone, you might need to adjust the Python path.
try:
    from src.data_models import Article
    from src.providers.fixture import FixtureProvider
    from scripts.demo import fetch_and_setup_data
except ImportError:
    # A simple Article class for standalone demonstration if imports fail
    from collections import namedtuple
    class Article:
        def __init__(self, id, title, url, description=None, content=None, source=None, published_at=None):
            self.id = id
            self.title = title
            self.url = url
            self.description = description
            self.content = content
            self.source = namedtuple("Source", ["name"])(name=source or "Unknown")
            self.published_at = published_at
    st.warning("Could not import local modules. Using fallback data structures. Functionality may be limited.")


# ---------------------------
# Helpers
# ---------------------------

def simple_summarize(article: Article, max_sentences: int = 3) -> str:
    """Simple extractive summarization using first sentences of content"""
    if not getattr(article, "content", None) or not isinstance(article.content, str):
        return (getattr(article, "description", "") or article.title)[:250]
    sentences = article.content.split(". ")
    summary = ". ".join(sentences[:max_sentences])
    if not summary.endswith("."):
        summary += "."
    return summary[:250]


def load_fixtures(folder: str, featured_limit: int, candidate_limit: int):
    """Load recent news articles from fixtures"""
    try:
        provider = FixtureProvider(folder=folder, shuffle=True)
        featured, candidates = asyncio.run(
            provider.fetch_featured_and_candidates(
                featured_limit=featured_limit, candidate_limit=candidate_limit
            )
        )
        all_articles = {a.id: a for a in (featured + candidates)}
        return featured, candidates, all_articles
    except Exception as e:
        st.error(f"Failed to load news fixtures: {e}. Please ensure the fixture files exist.")
        return [], [], {}


def _fixture_counts(folder: str) -> tuple[int, int, dict]:
    """Counts and metadata from fixtures"""
    try:
        p = pathlib.Path(folder)
        f = json.loads(p.joinpath("featured.json").read_text(encoding="utf-8"))
        pool = json.loads(p.joinpath("pool.json").read_text(encoding="utf-8"))
        featured_count = len(f.get("articles", f)) if isinstance(f, (dict, list)) else 0
        pool_count = len(pool.get("articles", pool)) if isinstance(pool, (dict, list)) else 0
        metadata = f.get("metadata", {}) if isinstance(f, dict) else {}
        return (featured_count, pool_count, metadata)
    except Exception:
        return (0, 0, {})


# ---- Modern Card-Grid Components ------------------------------------

def paginate(items: list, page_size: int, page_num: int):
    """Return slice for current page and total pages."""
    total = len(items)
    pages = max(1, (total + page_size - 1)) // page_size
    page_num = max(1, min(page_num, pages))
    start = (page_num - 1) * page_size
    end = start + page_size
    return items[start:end], pages, page_num


def create_article_grid(articles, cols_per_row, tab_id, rec_type, k, use_mmr):
    """Create a responsive grid of article cards"""
    for i in range(0, len(articles), cols_per_row):
        row_articles = articles[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for col, article in zip(cols, row_articles):
            with col:
                render_article_card(article, i + row_articles.index(article) + 1, tab_id, rec_type, k, use_mmr)


def render_article_card(article: Article, idx: int, tab_id: str, rec_type: str, k: int, use_mmr: bool):
    """Renders a card using native Streamlit elements."""
    unique_id = f"{tab_id}_{article.id}_{idx}"
    
    with st.container(border=True):
        meta_parts = []
        if hasattr(article, 'source') and article.source:
            meta_parts.append(getattr(article.source, "name", str(article.source)))
        if hasattr(article, 'published_at') and article.published_at:
            meta_parts.append(str(article.published_at)[:10])
        
        st.markdown(f"<p class='card-meta'>{' • '.join(meta_parts)}</p>", unsafe_allow_html=True)
        st.markdown(f"**{article.title}**")
        
        st.divider()
        
        c1, c2, c3 = st.columns(3)
        c1.link_button("Read →", article.url, use_container_width=True)

        if c2.button("🎯 Find Similar", key=f"rec_btn_{unique_id}", use_container_width=True):
            st.session_state.selected_article_id = article.id
            st.session_state.selected_article_title = article.title
            st.session_state.selected_article_url = article.url
            st.session_state.recommendation_type = rec_type
            st.session_state.num_recommendations = k
            st.session_state.use_diversity = use_mmr
            st.session_state.navigate_to = "🎯 AI Recommendations" # Set navigation intent
            st.rerun()

        in_basket = any(x["id"] == article.id for x in st.session_state.news_basket)
        basket_full = len(st.session_state.news_basket) >= 8
        
        with c3:
            if in_basket:
                st.button("✅ Added", key=f"added_btn_{unique_id}", disabled=True, use_container_width=True)
            elif basket_full:
                st.button("📦 Full", key=f"full_btn_{unique_id}", disabled=True, use_container_width=True)
            else:
                if st.button("📦 Save", key=f"save_btn_{unique_id}", use_container_width=True):
                    source_name = getattr(article.source, "name", "Unknown")
                    st.session_state.news_basket.append({
                        "id": article.id, "title": article.title, "url": article.url, "source": source_name,
                    })
                    st.rerun()


def create_recommendation_grid(recommendations, articles_dict, cols_per_row):
    """Create grid of recommendation cards"""
    for i in range(0, len(recommendations), cols_per_row):
        row_recs = recommendations[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for col, rec in zip(cols, row_recs):
            with col:
                # Use robust matching
                article = articles_dict.get(rec["title"].strip().lower())
                if article:
                    render_recommendation_card(rec, article, i + row_recs.index(rec) + 1)


def render_recommendation_card(rec: dict, article: Article, idx: int):
    """Modern recommendation card with a blue accent, using native elements."""
    unique_id = f"rec_card_{idx}_{article.id}"
    with st.container(border=True):
        st.markdown(f"<div class='card-accent' style='background-color: #3B82F6;'></div>", unsafe_allow_html=True)
        st.markdown(f"<p class='card-meta'>SCORE: {rec.get('score', 'N/A')}</p>", unsafe_allow_html=True)
        st.markdown(f"**{rec['title']}**")

        st.divider()
        c1, c2 = st.columns(2)
        c1.link_button("Read →", article.url, use_container_width=True)
        
        with c2:
            in_basket = any(item["id"] == article.id for item in st.session_state.news_basket)
            basket_full = len(st.session_state.news_basket) >= 8
            if in_basket:
                st.button("✅ Added", key=f"rec_in_basket_{unique_id}", disabled=True, use_container_width=True)
            elif basket_full:
                st.button("📦 Full", key=f"rec_full_{unique_id}", disabled=True, use_container_width=True)
            else:
                if st.button("📦 Save", key=f"rec_add_{unique_id}", use_container_width=True):
                    source_name = getattr(article.source, 'name', 'Unknown')
                    st.session_state.news_basket.append({
                        "id": article.id, "title": article.title, "url": article.url, "source": str(source_name),
                    })
                    st.rerun()


def create_summary_grid(articles, cols_per_row):
    """Create grid of summary cards"""
    for i in range(0, len(articles), cols_per_row):
        row_articles = articles[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for col, article in zip(cols, row_articles):
            with col:
                render_summary_card(article, i + row_articles.index(article) + 1)


def render_summary_card(article: Article, idx: int):
    """Modern summary card with a yellow accent, using native elements."""
    unique_id = f"summ_card_{idx}_{article.id}"
    with st.container(border=True):
        st.markdown(f"<div class='card-accent' style='background-color: #F59E0B;'></div>", unsafe_allow_html=True)
        st.markdown(f"**{article.title}**")

        st.divider()
        c1, c2 = st.columns(2)
        c1.link_button("Read →", article.url, use_container_width=True)
        if c2.button("🗑️ Remove", key=f"remove_summ_{unique_id}", use_container_width=True):
            st.session_state.news_basket = [x for x in st.session_state.news_basket if x["id"] != article.id]
            st.rerun()

# ---------------------------
# Main App
# ---------------------------

def main():
    st.set_page_config(page_title="AI News Dashboard", page_icon="🤖", layout="wide")

    ss = st.session_state
    ss.setdefault("news_basket", [])
    ss.setdefault("selected_article_id", None)
    ss.setdefault("selected_article_title", None)
    ss.setdefault("selected_article_url", None)
    ss.setdefault("recommendation_type", "Basic")
    ss.setdefault("num_recommendations", 5)
    ss.setdefault("use_diversity", True)
    ss.setdefault("articles_page", 1)
    ss.setdefault("recs_page", 1)
    ss.setdefault("summ_page", 1)
    ss.setdefault("active_tab", "📰 Featured Articles")

    # Handle programmatic navigation from button clicks before rendering any widgets
    if "navigate_to" in ss:
        ss.active_tab = ss.navigate_to
        del ss.navigate_to # Consume the navigation flag

    st.markdown("""
    <style>
      .block-container { padding: 2rem 3rem 3rem 3rem; }
      h1 { text-align: center; color: #111827; font-weight: 300; margin-bottom: 0.5rem; }
      
      /* Radio button styled as tabs */
      div[role="radiogroup"] {
          flex-direction: row;
          justify-content: center;
          margin-bottom: 2.5rem;
          border-bottom: 2px solid #F3F4F6;
      }
      div[role="radiogroup"] label {
          font-size: 17px;
          font-weight: 600;
          color: #6B7280;
          padding: 14px 24px;
          border-bottom: 2px solid transparent;
          transition: all 0.2s ease-in-out;
          border-radius: 0;
          margin: 0;
      }
      div[role="radiogroup"] input[type="radio"]:checked + div {
          color: #3B82F6;
          border-bottom: 2px solid #3B82F6;
      }

      .stButton > button { border-radius: 8px; font-weight: 600; transition: all 0.15s ease-in-out; }
      .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
      
      /* Card Styling using st.container */
      div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] {
          background-color: white;
          border-radius: 12px;
          border: 1px solid #E5E7EB;
          box-shadow: 0 1px 3px rgba(0,0,0,0.03), 0 1px 2px rgba(0,0,0,0.06);
          transition: all 0.2s ease-in-out;
          padding: 1.5rem 1.5rem 1rem 1.5rem;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          height: 100%; /* Make cards in a row equal height */
      }
      div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"]:hover {
          transform: translateY(-4px);
          box-shadow: 0 10px 15px -3px rgba(0,0,0,0.07), 0 4px 6px -2px rgba(0,0,0,0.05);
      }
      .card-accent {
          position: absolute; top: 0; left: 0; right: 0; height: 4px; border-top-left-radius: 12px; border-top-right-radius: 12px;
      }
      p.card-meta { color: #6B7280; font-size: 0.8rem; margin-bottom: 0.5rem; }
      
    </style>
    """, unsafe_allow_html=True)

    fixtures_folder = "src/providers/news_fixtures"
    featured_limit = 20
    candidate_limit = 100
    cur_feat, cur_pool, metadata = _fixture_counts(fixtures_folder)

    with st.sidebar:
        st.markdown("## 📊 System Status")
        col1, col2 = st.columns(2)
        col1.metric("Featured", cur_feat)
        col2.metric("Pool", cur_pool)
        if metadata.get("last_updated"):
            st.success(f"✅ Updated: {metadata['last_updated'][:16].replace('T', ' ')}")
        else:
            st.warning("⚠️ No timestamp")

        st.markdown("---")
        if st.button("🔄 Refresh News", type="primary", use_container_width=True):
            with st.spinner("Fetching latest news and building index..."):
                try:
                    success = fetch_and_setup_data(featured_count=featured_limit, pool_count=candidate_limit)
                    if success: st.success("✅ News refreshed!"); st.rerun()
                    else: st.error("❌ Update failed")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")
        st.markdown("## 🧺 News Basket")
        st.metric("Items", f"{len(ss.news_basket)}/8")
        if ss.news_basket and st.button("🗑️ Clear Basket", use_container_width=True):
            ss.news_basket = []; st.rerun()

    featured, candidates, all_articles = load_fixtures(fixtures_folder, featured_limit, candidate_limit)

    st.title("🤖 AI-Powered News Dashboard")
    st.markdown("<p style='text-align: center; color: #6b7280; font-size: 18px;'>Advanced recommendations with neural reranking and multi-model fusion</p>", unsafe_allow_html=True)

    with st.expander("⚙️ Configuration", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        ss.recommendation_type = c1.selectbox("AI Model", ["Basic", "Enhanced (Neural)", "Multi-Model"], index=["Basic", "Enhanced (Neural)", "Multi-Model"].index(ss.recommendation_type))
        ss.num_recommendations = c2.slider("Recommendations", 3, 15, ss.num_recommendations)
        ss.use_diversity = c3.checkbox("Diversity", value=ss.use_diversity)
        grid_columns = c4.selectbox("Grid Columns", [2, 3], index=0)

    # --- Tab implementation using styled radio buttons ---
    tab_options = ["📰 Featured Articles", "🎯 AI Recommendations", "📋 My Summaries"]
    try:
        current_tab_index = tab_options.index(ss.active_tab)
    except ValueError:
        current_tab_index = 0
    
    selected_tab = st.radio(
        "Main navigation", 
        tab_options, 
        index=current_tab_index, 
        horizontal=True, 
        label_visibility="collapsed"
    )

    if selected_tab != ss.active_tab:
        ss.active_tab = selected_tab
        st.rerun()


    if ss.active_tab == "📰 Featured Articles":
        if not featured:
            st.info("📭 No articles found. Click 'Refresh News' in the sidebar to fetch data.")
        else:
            items_per_page = grid_columns * 4 
            page_items, total_pages, ss.articles_page = paginate(featured, items_per_page, ss.articles_page)
            create_article_grid(page_items, grid_columns, "featured", ss.recommendation_type, ss.num_recommendations, ss.use_diversity)
            if total_pages > 1:
                st.divider()
                pg_cols = st.columns([1, 2, 1])
                if pg_cols[0].button("← Previous", disabled=ss.articles_page <= 1, use_container_width=True): ss.articles_page -= 1; st.rerun()
                pg_cols[1].markdown(f"<div style='text-align: center; margin-top: 5px;'>Page {ss.articles_page} of {total_pages}</div>", unsafe_allow_html=True)
                if pg_cols[2].button("Next →", disabled=ss.articles_page >= total_pages, use_container_width=True): ss.articles_page += 1; st.rerun()

    elif ss.active_tab == "🎯 AI Recommendations":
        if not ss.selected_article_id:
            st.info("🎯 Select an article and click 'Find Similar' to see AI-powered recommendations.")
        else:
            st.success(f"**Recommendations for:** [{ss.selected_article_title}]({ss.selected_article_url})")
            
            cache_key = f"recs_{ss.selected_article_id}_{ss.recommendation_type}_{ss.num_recommendations}"
            if cache_key not in ss:
                with st.spinner(f"Generating {ss.recommendation_type.lower()} recommendations..."):
                    cmd_map = { "Basic": ["--recommend", ss.selected_article_id], "Enhanced (Neural)": ["--enhanced", ss.selected_article_id], "Multi-Model": ["--multi-model", ss.selected_article_id], }
                    cmd = [sys.executable, "scripts/demo.py"] + cmd_map[ss.recommendation_type]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=os.getcwd(), env=os.environ.copy())
                    
                    recommendations = []
                    if result.returncode == 0 and result.stdout:
                        lines = result.stdout.strip().split("\n")
                        for line in lines:
                            if "|" not in line: continue
                            parts = [p.strip() for p in line.split("|")]
                            score = parts[0].split()[-1] if ' ' in parts[0] else parts[0]
                            recommendations.append({"title": parts[1], "score": score, "explanation": parts[2] if len(parts) > 2 else ""})
                    ss[cache_key] = recommendations

            rec_list = ss.get(cache_key, [])
            # Filter out the source article from its own recommendations
            rec_list = [rec for rec in rec_list if rec["title"].strip().lower() != ss.selected_article_title.strip().lower()]

            if rec_list:
                # Build a robust title-matching dictionary
                title_to_article = {a.title.strip().lower(): a for a in (featured + candidates)}
                
                items_per_page = grid_columns * 4
                page_recs, total_pages, ss.recs_page = paginate(rec_list, items_per_page, ss.recs_page)
                create_recommendation_grid(page_recs, title_to_article, grid_columns)
                if total_pages > 1:
                    st.divider()
                    pg_cols = st.columns([1, 2, 1])
                    if pg_cols[0].button("← Previous", key="rec_prev", disabled=ss.recs_page <= 1, use_container_width=True): ss.recs_page -= 1; st.rerun()
                    pg_cols[1].markdown(f"<div style='text-align: center; margin-top: 5px;'>Page {ss.recs_page} of {total_pages}</div>", unsafe_allow_html=True)
                    if pg_cols[2].button("Next →", key="rec_next", disabled=ss.recs_page >= total_pages, use_container_width=True): ss.recs_page += 1; st.rerun()
            else:
                st.warning("Could not generate recommendations for this article.")

    elif ss.active_tab == "📋 My Summaries":
        if not ss.news_basket:
            st.info("🧺 Your basket is empty. Save articles from other tabs to see summaries here.")
        else:
            basket_articles = [all_articles.get(item["id"]) for item in ss.news_basket if item["id"] in all_articles]
            items_per_page = grid_columns * 4
            page_articles, total_pages, ss.summ_page = paginate(basket_articles, items_per_page, ss.summ_page)
            create_summary_grid(page_articles, grid_columns)
            if total_pages > 1:
                st.divider()
                pg_cols = st.columns([1, 2, 1])
                if pg_cols[0].button("← Previous", key="summ_prev", disabled=ss.summ_page <= 1, use_container_width=True): ss.summ_page -= 1; st.rerun()
                pg_cols[1].markdown(f"<div style='text-align: center; margin-top: 5px;'>Page {ss.summ_page} of {total_pages}</div>", unsafe_allow_html=True)
                if pg_cols[2].button("Next →", key="summ_next", disabled=ss.summ_page >= total_pages, use_container_width=True): ss.summ_page += 1; st.rerun()


if __name__ == "__main__":
    main()

