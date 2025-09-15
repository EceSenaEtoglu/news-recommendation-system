import os
import asyncio
import streamlit as st
import subprocess, sys, json, pathlib
import uuid

# Assume src.data_models and other local modules are available in the path
# If running this script standalone, you might need to adjust the Python path.
from src.data_models import Article
from src.providers.fixture import FixtureProvider
from scripts.demo import fetch_and_setup_data

# Helpers
# Import summarization model
from src.summarization import summarize_article

def simple_summarize(article: Article, max_sentences: int = 3) -> str:
    """Enhanced summarization using transformer model with fallback"""
    if not getattr(article, "content", None) or not isinstance(article.content, str):
        return (getattr(article, "description", "") or article.title)[:350]
    
    # Use the new summarization model
    try:
        summary = summarize_article(
            content=article.content,
            title=article.title,
            max_length=300
        )
        return summary
    except Exception as e:
        # Fallback to simple extractive summarization
        print(f"Summarization failed: {e}")
        sentences = article.content.split(". ")
        summary = ". ".join(sentences[:max_sentences])
        if not summary.endswith("."):
            summary += "."
        return summary


# TODO, for the uknown category a category extraction model can be used
def get_category_badge(article: Article) -> str:
    """Get a styled category badge for an article"""
    category = None
    if hasattr(article, 'topics') and article.topics:
        category = article.topics[0]  # First topic is usually the category
    elif hasattr(article, 'source') and hasattr(article.source, 'category'):
        category = article.source.category
    
    if not category:
        return ""
    
    # Create a styled category badge
    category_colors = {
        'tech': '#3B82F6',         # Blue
        'technology': '#3B82F6',    # Blue (same as tech)
        'world': '#10B981',        # Green  
        'business': '#F59E0B',     # Orange
        'politics': '#EF4444',     # Red
        'science': '#8B5CF6',      # Purple
        'general': '#6B7280'       # Gray
    }
    color = category_colors.get(category.lower(), '#E5E7EB')  # Light gray for unknown categories
    return f"<span style='background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;'>{category.upper()}</span>"


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
    """Get counts from fixture metadata"""
    try:
        with open(f"{folder}/featured.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            metadata = data.get("metadata", {})
            return (
                metadata.get("featured_count", 0),
                metadata.get("pool_count", 0),
                metadata
            )
    except Exception as e:
        print(f"Error reading fixture counts: {e}")
        return 0, 0, {}


def paginate(items: list, items_per_page: int, current_page: int) -> tuple[list, int, int]:
    """Paginate items and return (page_items, total_pages, adjusted_page)"""
    if not items:
        return [], 0, 1
    
    total_pages = max(1, (len(items) + items_per_page - 1) // items_per_page)
    current_page = max(1, min(current_page, total_pages))
    
    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_items = items[start_idx:end_idx]
    
    return page_items, total_pages, current_page


def create_article_grid(articles: list, cols_per_row: int, tab_id: str, rec_type: str, k: int, use_mmr: bool):
    """Create grid of article cards"""
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
            # Format date and time
            pub_date = article.published_at
            if hasattr(pub_date, 'strftime'):
                date_str = pub_date.strftime("%Y-%m-%d")
                time_str = pub_date.strftime("%H:%M UTC")
                meta_parts.append(f"{date_str} {time_str}")
            else:
                meta_parts.append(str(pub_date)[:10])
        
        # Add category label if available
        category_badge = get_category_badge(article)
        if category_badge:
            st.markdown(category_badge, unsafe_allow_html=True)
        
        st.markdown(f"<p class='card-meta'>{' • '.join(meta_parts)}</p>", unsafe_allow_html=True)
        st.markdown(f"**{article.title}**")
        
        st.divider()
        
        c1, c2, c3 = st.columns(3)
        
        # Read button
        c1.link_button("Read →", article.url, use_container_width=True)

        # Find Similar button
        if c2.button("🎯 Find Similar", key=f"rec_btn_{unique_id}", use_container_width=True):
            st.session_state.selected_article_id = article.id
            st.session_state.selected_article_title = article.title
            st.session_state.selected_article_url = article.url
            st.session_state.recommendation_type = rec_type
            st.session_state.num_recommendations = k
            st.session_state.use_diversity = use_mmr
            st.session_state.navigate_to = "🎯 AI Recommendations" # Set navigation intent
            st.rerun()

        # Save/Full button
        in_basket = any(x["id"] == article.id for x in st.session_state.news_basket)
        if in_basket:
            c3.button("📦 Full", key=f"full_btn_{unique_id}", disabled=True, use_container_width=True)
        else:
            if c3.button("📦 Save", key=f"save_btn_{unique_id}", use_container_width=True):
                source_name = getattr(article.source, "name", "Unknown")
                st.session_state.news_basket.append({
                    "id": article.id, "title": article.title, "url": article.url, "source": source_name,
                })
                st.rerun()


def create_recommendation_grid(recommendations, articles_dict, cols_per_row, recommendation_type="Basic"):
    """Create grid of recommendation cards"""
    for i in range(0, len(recommendations), cols_per_row):
        row_recs = recommendations[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for col, rec in zip(cols, row_recs):
            with col:
                # Try to find the article for additional metadata, but use recommendation data as primary
                rec_title = rec["title"].strip().lower()
                article = articles_dict.get(rec_title)
                
                # If we have a URL in the recommendation, use it directly
                if rec.get("url") and rec["url"] != "#":
                    # Create article object from recommendation data
                    from collections import namedtuple
                    Source = namedtuple("Source", ["name"])
                    Article = namedtuple("Article", ["id", "title", "url", "description", "source", "published_at", "topics"])
                    
                    # Use source from recommendation data
                    source_name = rec.get("source", "Unknown")
                    
                    # Try to get additional metadata from matched article
                    if article:
                        recommendation_article = Article(
                            id=article.id,
                            title=rec["title"],
                            url=rec["url"],
                            description=getattr(article, "description", ""),
                            source=Source(name=source_name),
                            published_at=getattr(article, "published_at", None),
                            topics=getattr(article, "topics", [])
                        )
                    else:
                        # Fallback with minimal data
                        recommendation_article = Article(
                            id=rec.get("id", "unknown"),
                            title=rec["title"],
                            url=rec["url"],
                            description=rec.get("description", ""),
                            source=Source(name=source_name),
                            published_at=rec.get("published_at", None),
                            topics=rec.get("topics", [])
                        )
                    
                    render_recommendation_card(rec, recommendation_article, i + row_recs.index(rec) + 1, recommendation_type)
                else:
                    # Fallback to old matching logic if no URL provided
                    if not article:
                        # Try fuzzy matching strategies
                        best_match = None
                        best_score = 0
                        
                        for title, art in articles_dict.items():
                            # Strategy 1: One title contains the other
                            if rec_title in title or title in rec_title:
                                article = art
                                break
                            
                            # Strategy 2: Word overlap scoring
                            rec_words = set(rec_title.split())
                            title_words = set(title.split())
                            if rec_words and title_words:
                                overlap = len(rec_words.intersection(title_words))
                                total_words = len(rec_words.union(title_words))
                                score = overlap / total_words if total_words > 0 else 0
                                
                                if score > best_score and score > 0.3:  # At least 30% word overlap
                                    best_match = art
                                    best_score = score
                        
                        if not article and best_match:
                            article = best_match
                    
                    if article:
                        render_recommendation_card(rec, article, i + row_recs.index(rec) + 1, recommendation_type)
                    else:
                        # Fallback: create a minimal article object from recommendation data
                        from collections import namedtuple
                        Source = namedtuple("Source", ["name"])
                        Article = namedtuple("Article", ["id", "title", "url", "description", "source", "published_at", "topics"])
                        fallback_article = Article(
                            id=rec.get("id", "unknown"),
                            title=rec["title"],
                            url=rec.get("url", "#"),
                            description=rec.get("description", ""),
                            source=Source(name=rec.get("source", "Unknown")),
                            published_at=rec.get("published_at", None),
                            topics=rec.get("topics", [])
                        )
                        render_recommendation_card(rec, fallback_article, i + row_recs.index(rec) + 1, recommendation_type)


def render_recommendation_card(rec: dict, article: Article, idx: int, recommendation_type: str = "Basic"):
    """Modern recommendation card with a blue accent, using native elements."""
    unique_id = f"rec_card_{idx}_{article.id}"
    with st.container(border=True):
        st.markdown(f"<div class='card-accent' style='background-color: #3B82F6;'></div>", unsafe_allow_html=True)
        
        # Add category label if available
        category_badge = get_category_badge(article)
        if category_badge:
            st.markdown(category_badge, unsafe_allow_html=True)
        
        # Add source and time metadata like in featured articles
        meta_parts = []
        if hasattr(article, 'source') and article.source:
            meta_parts.append(getattr(article.source, "name", str(article.source)))
        if hasattr(article, 'published_at') and article.published_at:
            # Format date and time
            pub_date = article.published_at
            if hasattr(pub_date, 'strftime'):
                date_str = pub_date.strftime("%Y-%m-%d")
                time_str = pub_date.strftime("%H:%M UTC")
                meta_parts.append(f"{date_str} {time_str}")
            else:
                meta_parts.append(str(pub_date)[:10])
        
        if meta_parts:
            st.markdown(f"<p class='card-meta'>{' • '.join(meta_parts)}</p>", unsafe_allow_html=True)
        
        score = rec.get('score', 'N/A')
        if isinstance(score, (int, float)):
            # Add score range information based on recommendation type
            if recommendation_type == "Basic":
                score_info = f"SCORE: {score:.3f} (0.0-1.0+ semantic similarity)"
            elif recommendation_type == "Enhanced (Neural)":
                score_info = f"SCORE: {score:.3f} (0.0-1.0+ neural reranked)"
            elif recommendation_type == "Multi-Model":
                score_info = f"SCORE: {score:.3f} (0.0-1.0+ multi-model fusion)"
            else:
                score_info = f"SCORE: {score:.3f}"
        else:
            score_info = f"SCORE: {score}"
        st.markdown(f"<p class='card-meta'>{score_info}</p>", unsafe_allow_html=True)
        st.markdown(f"**{rec['title']}**")

        st.divider()
        c1, c2 = st.columns(2)
        c1.link_button("Read →", article.url, use_container_width=True)
        
        # Save/Full button - check if already in basket
        in_basket = any(x["id"] == article.id for x in st.session_state.news_basket)
        if in_basket:
            c2.button("📦 Full", key=f"full_rec_{unique_id}", disabled=True, use_container_width=True)
        else:
            if c2.button("📦 Save", key=f"save_rec_{unique_id}", use_container_width=True):
                source_name = getattr(article.source, "name", "Unknown")
                st.session_state.news_basket.append({
                    "id": article.id, "title": article.title, "url": article.url, "source": source_name,
                })
                st.rerun()


def create_saved_articles_grid(articles: list, cols_per_row: int):
    """Create grid of saved article cards"""
    for i in range(0, len(articles), cols_per_row):
        row_articles = articles[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for col, article in zip(cols, row_articles):
            with col:
                render_saved_article_card(article, i + row_articles.index(article) + 1)


def render_saved_article_card(article: Article, idx: int):
    """Card for saved articles with a summarize button."""
    unique_id = f"saved_card_{idx}_{article.id}"
    with st.container(border=True):
        st.markdown(f"<div class='card-accent' style='background-color: #F59E0B;'></div>", unsafe_allow_html=True)
        
        # Add category label if available
        category_badge = get_category_badge(article)
        if category_badge:
            st.markdown(category_badge, unsafe_allow_html=True)
        
        # Add source and time metadata like in featured articles
        meta_parts = []
        if hasattr(article, 'source') and article.source:
            meta_parts.append(getattr(article.source, "name", str(article.source)))
        if hasattr(article, 'published_at') and article.published_at:
            # Format date and time
            pub_date = article.published_at
            if hasattr(pub_date, 'strftime'):
                date_str = pub_date.strftime("%Y-%m-%d")
                time_str = pub_date.strftime("%H:%M UTC")
                meta_parts.append(f"{date_str} {time_str}")
            else:
                meta_parts.append(str(pub_date)[:10])
        
        if meta_parts:
            st.markdown(f"<p class='card-meta'>{' • '.join(meta_parts)}</p>", unsafe_allow_html=True)
        
        st.markdown(f"**{article.title}**")

        # Conditional Summary Display
        if st.session_state.get("summarize_id") == article.id:
            st.caption(simple_summarize(article))
            st.divider()
            c1, c2 = st.columns(2)
            c1.link_button("Read →", article.url, use_container_width=True)
            if c2.button("Hide Summary", key=f"hide_summ_{unique_id}", use_container_width=True):
                st.session_state.summarize_id = None
                st.rerun()
        else:
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.link_button("Read →", article.url, use_container_width=True)
            if c2.button("📝 Summarize", key=f"summ_btn_{unique_id}", use_container_width=True):
                st.session_state.summarize_id = article.id
                st.rerun()
            if c3.button("🗑️ Remove", key=f"remove_btn_{unique_id}", use_container_width=True):
                st.session_state.news_basket = [x for x in st.session_state.news_basket if x["id"] != article.id]
                st.rerun()


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="AI News Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for modern styling
    st.markdown("""
    <style>
    .card-accent {
        height: 4px;
        margin: -1rem -1rem 1rem -1rem;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    .card-meta {
        font-size: 0.875rem;
        color: #6b7280;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    ss = st.session_state
    ss.setdefault("news_basket", [])
    ss.setdefault("selected_article_id", None)
    ss.setdefault("selected_article_title", None)
    ss.setdefault("selected_article_url", None)
    ss.setdefault("recommendation_type", "Multi-Model")
    ss.setdefault("num_recommendations", 5)
    ss.setdefault("use_diversity", True)
    ss.setdefault("articles_page", 1)
    ss.setdefault("recs_page", 1)
    ss.setdefault("summ_page", 1)
    ss.setdefault("active_tab", "📰 Featured Articles")
    ss.setdefault("summarize_id", None) # ID of article to summarize
    ss.setdefault("featured_count", 20)
    ss.setdefault("candidate_count", 100)

    # Handle programmatic navigation from button clicks before rendering any widgets
    if "navigate_to" in ss:
        ss.active_tab = ss.navigate_to
        del ss["navigate_to"]
        st.rerun()
        
    # Sidebar
    with st.sidebar:
        st.title("🤖 AI News")
        
        # System Status
        fixtures_folder = "src/providers/news_fixtures"
        featured_count, pool_count, metadata = _fixture_counts(fixtures_folder)
        
        st.markdown("### System Status")
        st.metric("Featured", featured_count)
        st.metric("Pool", pool_count)
        
        if metadata.get("last_updated"):
            # Parse the date and format it nicely
            from datetime import datetime, timezone
            try:
                last_updated = metadata['last_updated']
                
                # Handle different date formats
                if len(last_updated) == 10:  # YYYY-MM-DD format
                    date_str = last_updated
                    current_time = datetime.now(timezone.utc).strftime("%H:%M UTC")
                    st.success(f"Updated: {date_str} {current_time}")
                elif 'T' in last_updated:  # ISO format like 2025-09-15T08:32:19.388663+00:00
                    # Parse ISO datetime and format it nicely
                    dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    date_str = dt.strftime("%Y-%m-%d")
                    time_str = dt.strftime("%H:%M UTC")
                    st.success(f"Updated: {date_str} {time_str}")
                else:
                    # Fallback for other formats
                    st.success(f"Updated: {last_updated}")
            except Exception as e:
                # If parsing fails, show the original string
                st.success(f"Updated: {metadata['last_updated']}")
        
        st.markdown("### Fetch News Config")
        ss.featured_count = st.slider("Featured Articles", 10, 50, ss.featured_count)
        ss.candidate_count = st.slider("Candidate Pool", 50, 200, ss.candidate_count)

        if st.button("🔄 Refresh News", type="primary", use_container_width=True):
            with st.spinner("Fetching latest news and building index... This can take a short while."):
                try:
                    success = fetch_and_setup_data(featured_count=ss.featured_count, pool_count=ss.candidate_count)
                    if success: st.success("✅ News refreshed!"); st.rerun()
                    else: st.error("❌ Update failed")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        st.markdown("### News Basket")
        st.metric("Items", f"{len(ss.news_basket)}/8")
        
        if ss.news_basket and st.button("🗑️ Clear Basket", use_container_width=True):
            ss.news_basket = []; st.rerun()

    featured, candidates, all_articles = load_fixtures(fixtures_folder, ss.featured_count, ss.candidate_count)
    

    st.title("🤖 AI-Powered News Dashboard")
    st.markdown("<p style='text-align: center; color: #6b7280; font-size: 18px;'>Advanced recommendations with neural reranking and multi-model fusion</p>", unsafe_allow_html=True)

    with st.expander("⚙️ Recommender Configs", expanded=False):
        c1, c2, c3 = st.columns(3)
        ss.recommendation_type = c1.selectbox("AI Model", ["Basic", "Enhanced (Neural)", "Multi-Model"], index=["Basic", "Enhanced (Neural)", "Multi-Model"].index(ss.recommendation_type))
        ss.num_recommendations = c2.slider("Recommendations", 3, 15, ss.num_recommendations)
        ss.use_diversity = c3.checkbox("Diversity", value=ss.use_diversity)
        
        # Add score explanation
        if ss.recommendation_type == "Basic":
            st.info("📊 **Basic**: Semantic similarity scores (0.0-1.0+) based on cosine similarity between article embeddings. Higher scores indicate more similar content.")
        elif ss.recommendation_type == "Enhanced (Neural)":
            st.info("🧠 **Enhanced**: Neural reranked scores (0.0-1.0+) using a trained neural network that considers semantic similarity, topic overlap, recency, and content quality.")
        elif ss.recommendation_type == "Multi-Model":
            st.info("🔄 **Multi-Model**: Fusion scores (0.0-1.0+) combining multiple embedding models using weighted averaging. News-specific models get higher weights.")

    # --- Tab implementation using styled radio buttons ---
    tab_cols = st.columns(3)
    with tab_cols[0]:
        if st.button("📰 Featured Articles", key="tab_featured", use_container_width=True, type="primary" if ss.active_tab == "📰 Featured Articles" else "secondary"):
            ss.active_tab = "📰 Featured Articles"; st.rerun()
    with tab_cols[1]:
        if st.button("🎯 AI Recommendations", key="tab_recs", use_container_width=True, type="primary" if ss.active_tab == "🎯 AI Recommendations" else "secondary"):
            ss.active_tab = "🎯 AI Recommendations"; st.rerun()
    with tab_cols[2]:
        if st.button("📚 Saved Articles", key="tab_saved", use_container_width=True, type="primary" if ss.active_tab == "📚 Saved Articles" else "secondary"):
            ss.active_tab = "📚 Saved Articles"; st.rerun()

    # --- Tab Content ---
    if ss.active_tab == "📰 Featured Articles":
        if not featured:
            st.info("📭 No articles found. Click 'Refresh News' in the sidebar to fetch data.")
        else:
            items_per_page = 2 * 4  # 2 columns * 4 rows = 8 items per page
            page_items, total_pages, ss.articles_page = paginate(featured, items_per_page, ss.articles_page)
            create_article_grid(page_items, 2, "featured", ss.recommendation_type, ss.num_recommendations, ss.use_diversity)
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
                    cmd = [sys.executable, "scripts/demo.py"] + cmd_map[ss.recommendation_type] + ["--k", str(ss.num_recommendations)]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=os.getcwd(), env=os.environ.copy())
                    
                    recommendations = []
                    if result.returncode == 0 and result.stdout:
                        lines = result.stdout.strip().split("\n")
                        current_rec = None
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                                
                            # Look for lines that start with a number followed by a dot (e.g., "1. 0.816 | ...")
                            if line[0].isdigit() and "." in line and "|" in line:
                                # Save previous recommendation if exists
                                if current_rec:
                                    recommendations.append(current_rec)
                                
                                # Parse new recommendation
                                parts = line.split("|")
                                if len(parts) >= 2:
                                    # Extract score from the first part (e.g., "1. 0.816" -> "0.816")
                                    score_part = parts[0].strip()
                                    if ". " in score_part:
                                        score_str = score_part.split(". ", 1)[1].strip()
                                    else:
                                        score_str = score_part
                                    
                                    # Parse title and URL (new format: "title | url")
                                    if len(parts) >= 3:
                                        # New format: "1. 0.816 | title | url"
                                        title = parts[1].strip()
                                        url = parts[2].strip()
                                    else:
                                        # Old format: "1. 0.816 | title"
                                        title = parts[1].strip()
                                        url = "#"
                                    
                                    try:
                                        score = float(score_str)
                                        current_rec = {"title": title, "score": score, "url": url, "source": "Unknown"}
                                    except:
                                        current_rec = {"title": title, "score": 0.0, "url": url, "source": "Unknown"}
                            
                            # Look for source information (e.g., "   Source: www.cnbc.com")
                            elif line.startswith("   Source: ") and current_rec:
                                source = line.replace("   Source: ", "").strip()
                                current_rec["source"] = source
                        
                        # Add the last recommendation
                        if current_rec:
                            recommendations.append(current_rec)
                    
                    ss[cache_key] = recommendations
            
            rec_list = ss.get(cache_key, [])
            
            if rec_list:
                # Build a robust title-matching dictionary
                title_to_article = {a.title.strip().lower(): a for a in (featured + candidates)}
                
                items_per_page = 2 * 4  # 2 columns * 4 rows = 8 items per page
                page_recs, total_pages, ss.recs_page = paginate(rec_list, items_per_page, ss.recs_page)
                create_recommendation_grid(page_recs, title_to_article, 2, ss.recommendation_type)
                if total_pages > 1:
                    st.divider()
                    pg_cols = st.columns([1, 2, 1])
                    if pg_cols[0].button("← Previous", key="rec_prev", disabled=ss.recs_page <= 1, use_container_width=True): ss.recs_page -= 1; st.rerun()
                    pg_cols[1].markdown(f"<div style='text-align: center; margin-top: 5px;'>Page {ss.recs_page} of {total_pages}</div>", unsafe_allow_html=True)
                    if pg_cols[2].button("Next →", key="rec_next", disabled=ss.recs_page >= total_pages, use_container_width=True): ss.recs_page += 1; st.rerun()
            else:
                st.warning("No recommendations found. Try selecting a different article.")

    elif ss.active_tab == "📚 Saved Articles":
        if not ss.news_basket:
            st.info("🧺 Your basket is empty. Save articles from other tabs to see them here.")
        else:
            
            basket_articles = []
            for item in ss.news_basket:
                article = all_articles.get(item["id"])
                if article:
                    basket_articles.append(article)
                else:
                    st.warning(f"Article not found in loaded data: {item['title']} (ID: {item['id']})")
            
            
            if basket_articles:
                items_per_page = 2 * 4  # 2 columns * 4 rows = 8 items per page
                page_articles, total_pages, ss.summ_page = paginate(basket_articles, items_per_page, ss.summ_page)
                create_saved_articles_grid(page_articles, 2)
            else:
                st.warning("No articles found in loaded data. Try refreshing the news data.")
            
            if basket_articles and total_pages > 1:
                st.divider()
                pg_cols = st.columns([1, 2, 1])
                if pg_cols[0].button("← Previous", key="summ_prev", disabled=ss.summ_page <= 1, use_container_width=True): ss.summ_page -= 1; st.rerun()
                pg_cols[1].markdown(f"<div style='text-align: center; margin-top: 5px;'>Page {ss.summ_page} of {total_pages}</div>", unsafe_allow_html=True)
                if pg_cols[2].button("Next →", key="summ_next", disabled=ss.summ_page >= total_pages, use_container_width=True): ss.summ_page += 1; st.rerun()


if __name__ == "__main__":
    main()
