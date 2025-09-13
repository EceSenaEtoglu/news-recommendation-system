import os
import asyncio
import streamlit as st
import subprocess, sys, json, pathlib

from src.providers.fixture import FixtureProvider
from scripts.demo import fetch_and_setup_data
from src.data_models import Article

def simple_summarize(article: Article, max_sentences: int = 3) -> str:
    """Simple extractive summarization using first sentences of content"""
    if not article.content:
        return article.description or article.title
    
    # Split into sentences and take first few
    sentences = article.content.split('. ')
    summary_sentences = sentences[:max_sentences]
    summary = '. '.join(summary_sentences)
    
    # Ensure it ends with a period
    if not summary.endswith('.'):
        summary += '.'
    
    return summary

# loads the recent news articles
def load_fixtures(folder: str, featured_limit: int, candidate_limit: int):
    provider = FixtureProvider(folder=folder, shuffle=True)
    featured, candidates = asyncio.run(
        provider.fetch_featured_and_candidates(
            featured_limit=featured_limit, candidate_limit=candidate_limit
        )
    )
    # Build map for quick lookups
    all_articles = {a.id: a for a in (featured + candidates)}
    return featured, candidates, all_articles


# counts the number of featured and candidate articles and gets metadata
def _fixture_counts(folder: str) -> tuple[int, int, dict]:
    try:
        p = pathlib.Path(folder)
        f = json.loads(p.joinpath("featured.json").read_text(encoding="utf-8"))
        pool = json.loads(p.joinpath("pool.json").read_text(encoding="utf-8"))
        
        # Handle new structure with metadata
        featured_count = 0
        pool_count = 0
        metadata = {}
        
        if isinstance(f, dict) and "articles" in f:
            featured_count = len(f["articles"])
            if "metadata" in f:
                metadata = f["metadata"]
        elif isinstance(f, list):
            featured_count = len(f)
            
        if isinstance(pool, dict) and "articles" in pool:
            pool_count = len(pool["articles"])
        elif isinstance(pool, list):
            pool_count = len(pool)
            
        return (featured_count, pool_count, metadata)
    except Exception:
        return (0, 0, {})


def main():
    st.set_page_config(
        page_title="AI News Recommendations",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for news basket
    if 'news_basket' not in st.session_state:
        st.session_state.news_basket = []
    
    st.title("🤖 AI-Powered News Recommendations")
    st.markdown("Discover relevant articles using advanced AI models and neural networks")
    
    # Add some styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .article-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar config
    st.sidebar.header("📊 System Status")
    
    # Hidden technical settings (for internal use)
    fixtures_folder = "src/providers/news_fixtures"
    index_path = "db/faiss.index"
    meta_path = "db/faiss_metadata.pkl"
    featured_limit = 20
    candidate_limit = 100
    
    # Show current counts from selected folder
    cur_feat, cur_pool, metadata = _fixture_counts(fixtures_folder)
    
    # Status indicators
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Featured Articles", cur_feat)
    with col2:
        st.metric("Pool Articles", cur_pool)
    
    # Show last update time
    if metadata.get("last_updated"):
        from datetime import datetime
        try:
            last_update = datetime.fromisoformat(metadata["last_updated"].replace('Z', '+00:00'))
            formatted_time = last_update.strftime("%Y-%m-%d %H:%M UTC")
            st.sidebar.success(f"✅ Last updated: {formatted_time}")
        except Exception:
            st.sidebar.warning("⚠️ Last updated: Unknown")
    else:
        st.sidebar.warning("⚠️ No update timestamp found")

    # User-friendly controls
    st.sidebar.markdown("---")
    st.sidebar.header("🔄 Actions")
    
    if st.sidebar.button("📰 Get Most Recent News", type="primary"):
        with st.spinner("Fetching latest news and updating recommendations..."):
            try:
                # Call the function directly instead of subprocess
                success = fetch_and_setup_data(
                    featured_count=int(featured_limit),
                    pool_count=int(candidate_limit)
                )
                
                if success:
                    st.sidebar.success("🎉 News updated successfully!")
                    st.rerun()  # Refresh the page to show new articles
                else:
                    st.sidebar.error("❌ Failed to fetch and setup news")
                        
            except Exception as e:
                st.sidebar.error(f"Update failed: {e}")
    
    # Help section
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ How to Use")
    st.sidebar.markdown("""
    1. **Get News**: Click "Get Most Recent News" to fetch latest articles
    2. **Choose Settings**: Select AI model and recommendation count
    3. **Get Recommendations**: Click "🎯 Recommend" next to any article
    4. **View Results**: Recommendations appear below the article list
    """)
    
    # AI Model explanations
    with st.sidebar.expander("🤖 AI Models Explained"):
        st.markdown("""
        **Basic**: Traditional BM25 + semantic search
        
        **Enhanced (Neural)**: Adds neural reranker with advanced features
        
        **Multi-Model**: Combines multiple embedding models for better results
        """)
    
    # News Basket section
    st.sidebar.markdown("---")
    st.sidebar.header("📰 News Basket")
    
    basket_count = len(st.session_state.news_basket)
    st.sidebar.metric("Articles Selected", basket_count, f"Max: 8")
    
    if basket_count > 0:
        if st.sidebar.button("🗑️ Clear Basket"):
            st.session_state.news_basket = []
            st.rerun()
        
        if st.sidebar.button("📝 Generate Summaries", disabled=(basket_count == 0)):
            st.session_state.show_summaries = True
            st.rerun()
    
    if basket_count >= 8:
        st.sidebar.warning("⚠️ Basket is full (8/8)")

    # Load
    featured, candidates, all_articles = load_fixtures(
        fixtures_folder, int(featured_limit), int(candidate_limit)
    )

    # UI: Show featured articles with individual recommendation buttons
    st.subheader("📰 Latest News Articles")
    show_n = st.number_input("How many articles to show", min_value=1, max_value=100, value=15)
    
    if not featured:
        st.info("📰 No articles found. Click 'Get Most Recent News' to fetch the latest articles!")
    else:
        # Global recommendation settings
        st.markdown("**⚙️ Recommendation Settings**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rec_type = st.selectbox(
                "AI Model",
                ["Basic", "Enhanced (Neural)", "Multi-Model"],
                help="Enhanced uses neural reranker, Multi-Model uses fusion"
            )
        
        with col2:
            k = st.slider("Number of recommendations", min_value=3, max_value=20, value=5)
        
        with col3:
            use_mmr = st.checkbox("Enable diversity", value=True, help="Reduce similar recommendations")
        
        st.markdown("---")
        
        # Add a button to show recommendations if we have a selected article
        if hasattr(st.session_state, 'selected_article_id'):
            if st.button("🎯 Show Recommendations", key="show_recs_main"):
                # Recommendations will be shown in the column layout below
                pass
        
        # Display articles with individual recommendation buttons
        # Only show the main article list if no recommendations are currently displayed
        if not hasattr(st.session_state, 'selected_article_id'):
            for idx, a in enumerate(featured[: int(show_n)], start=1):
                # Create columns for article title and action buttons
                col_title, col_buttons = st.columns([3, 2])
                
                with col_title:
                    if a.url:
                        st.markdown(f"**{idx}.** [{a.title}]({a.url})")
                    else:
                        st.markdown(f"**{idx}.** {a.title}")
                    
                    # Show article metadata if available
                    if hasattr(a, 'source') and a.source:
                        st.caption(f"📰 {a.source}")
                    if hasattr(a, 'published_at') and a.published_at:
                        st.caption(f"🕒 {a.published_at}")
                
                with col_buttons:
                    col_rec, col_basket = st.columns(2)
                    
                    with col_rec:
                        if st.button(f"🎯", key=f"rec_{a.id}", help=f"Get recommendations for: {a.title[:50]}..."):
                            # Store the selected article ID in session state
                            st.session_state.selected_article_id = a.id
                            st.session_state.selected_article_title = a.title
                            st.session_state.recommendation_type = rec_type
                            st.session_state.num_recommendations = k
                            st.session_state.use_diversity = use_mmr
                            st.rerun()
                    
                    with col_basket:
                        # Check if article is already in basket
                        article_in_basket = any(item['id'] == a.id for item in st.session_state.news_basket)
                        basket_full = len(st.session_state.news_basket) >= 8
                        
                        if article_in_basket:
                            if st.button("✅", key=f"basket_{a.id}", help="Already in basket"):
                                # Remove from basket
                                st.session_state.news_basket = [item for item in st.session_state.news_basket if item['id'] != a.id]
                                st.rerun()
                        elif basket_full:
                            st.button("📦", key=f"basket_{a.id}", disabled=True, help="Basket is full (8/8)")
                        else:
                            if st.button("📦", key=f"basket_{a.id}", help=f"Add to basket: {a.title[:50]}..."):
                                # Add to basket
                                st.session_state.news_basket.append({
                                    'id': a.id,
                                    'title': a.title,
                                    'url': a.url,
                                    'source': getattr(a.source, 'name', 'Unknown') if hasattr(a, 'source') else 'Unknown'
                                })
                                st.rerun()
        
        # Show recommendations if an article was selected - use columns for better visibility
        if hasattr(st.session_state, 'selected_article_id'):
            # Create two columns: articles on left, recommendations on right
            col_articles, col_recommendations = st.columns([1, 1])
            
            with col_recommendations:
                st.subheader(f"🎯 Recommendations for:")
                st.markdown(f"**{st.session_state.selected_article_title}**")
                
                # Add a button to refresh recommendations
                if st.button("🔄 Refresh Recommendations", key="refresh_recs"):
                    # Clear cached recommendations to force refresh
                    if 'cached_recommendations' in st.session_state:
                        del st.session_state.cached_recommendations
                    st.rerun()
                
                # Get the recommendation type from session state
                current_rec_type = st.session_state.recommendation_type
                current_k = st.session_state.num_recommendations
                
                # Check if we have cached recommendations
                cache_key = f"recs_{st.session_state.selected_article_id}_{current_rec_type}_{current_k}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = None
                
                if st.session_state[cache_key] is None:
                    try:
                        with st.spinner(f"🤖 Getting {current_rec_type.lower()} recommendations..."):
                            # Choose command based on recommendation type
                            if current_rec_type == "Basic":
                                cmd = [sys.executable, "scripts/demo.py", "--recommend", st.session_state.selected_article_id]
                            elif current_rec_type == "Enhanced (Neural)":
                                cmd = [sys.executable, "scripts/demo.py", "--enhanced", st.session_state.selected_article_id]
                            elif current_rec_type == "Multi-Model":
                                cmd = [sys.executable, "scripts/demo.py", "--multi-model", st.session_state.selected_article_id]
                            
                            res = subprocess.run(
                                cmd, 
                                capture_output=True, 
                                text=True, 
                                check=False,
                                cwd=os.getcwd(),
                                env=os.environ.copy()
                            )
                        
                        if res.returncode == 0 and res.stdout:
                            # Parse and cache the recommendations
                            recommendations = []
                            lines = res.stdout.strip().split('\n')
                            for line in lines:
                                if '|' in line:
                                    parts = line.split('|')
                                    if len(parts) >= 2:
                                        # Extract the original number from the line (e.g., "6. 0.514 | title")
                                        original_num = parts[0].strip().split('.')[0] if '.' in parts[0] else "1"
                                        score = parts[0].strip().split('.')[1].strip() if '.' in parts[0] else parts[0].strip()
                                        title = parts[1].strip()
                                        explanation = parts[2].strip() if len(parts) > 2 else ""
                                        
                                        recommendations.append({
                                            'original_num': original_num,
                                            'score': score,
                                            'title': title,
                                            'explanation': explanation
                                        })
                            
                            st.session_state[cache_key] = recommendations
                        else:
                            st.warning("⚠️ No recommendations found. Try clicking 'Get Most Recent News' to update the article database.")
                            if res.stderr:
                                st.caption(f"Error: {res.stderr}")
                                
                    except Exception as e:
                        st.error(f"❌ Recommendation failed: {e}")
                
                # Display cached recommendations
                if st.session_state[cache_key]:
                    recommendations = st.session_state[cache_key]
                    for i, rec in enumerate(recommendations, 1):
                        # Look up the actual article to get URL
                        recommended_article = None
                        for article in (featured + candidates):
                            if article.title == rec['title']:
                                recommended_article = article
                                break
                        
                        # Create a nice card-like display with link and basket button
                        with st.container():
                            col_title, col_basket = st.columns([4, 1])
                            
                            with col_title:
                                if recommended_article and recommended_article.url:
                                    st.markdown(f"**{i}.** [{rec['title']}]({recommended_article.url})")
                                else:
                                    st.markdown(f"**{i}.** {rec['title']}")
                                
                                st.markdown(f"🎯 **Score:** {rec['score']}")
                                if rec['explanation']:
                                    st.caption(f"💡 {rec['explanation']}")
                                
                                # Add source info if available
                                if recommended_article and hasattr(recommended_article, 'source') and recommended_article.source:
                                    st.caption(f"📰 {recommended_article.source}")
                            
                            with col_basket:
                                if recommended_article:
                                    # Check if article is already in basket
                                    article_in_basket = any(item['id'] == recommended_article.id for item in st.session_state.news_basket)
                                    basket_full = len(st.session_state.news_basket) >= 8
                                    
                                    if article_in_basket:
                                        if st.button("✅", key=f"rec_basket_{recommended_article.id}", help="Already in basket"):
                                            # Remove from basket
                                            st.session_state.news_basket = [item for item in st.session_state.news_basket if item['id'] != recommended_article.id]
                                            st.rerun()
                                    elif basket_full:
                                        st.button("📦", key=f"rec_basket_{recommended_article.id}", disabled=True, help="Basket is full (8/8)")
                                    else:
                                        if st.button("📦", key=f"rec_basket_{recommended_article.id}", help=f"Add to basket: {rec['title'][:50]}..."):
                                            # Add to basket
                                            st.session_state.news_basket.append({
                                                'id': recommended_article.id,
                                                'title': recommended_article.title,
                                                'url': recommended_article.url,
                                                'source': getattr(recommended_article.source, 'name', 'Unknown') if hasattr(recommended_article, 'source') else 'Unknown'
                                            })
                                            st.rerun()
                            
                            st.markdown("---")
                
                # Add a button to clear recommendations
                if st.button("🗑️ Clear Recommendations"):
                    if 'selected_article_id' in st.session_state:
                        del st.session_state.selected_article_id
                    if 'selected_article_title' in st.session_state:
                        del st.session_state.selected_article_title
                    # Clear cached recommendations
                    for key in list(st.session_state.keys()):
                        if key.startswith('recs_'):
                            del st.session_state[key]
                    st.rerun()
            
            with col_articles:
                st.subheader("📰 All Articles")
                # Show a condensed list of all articles
                for idx, a in enumerate(featured[: int(show_n)], start=1):
                    if a.url:
                        st.markdown(f"{idx}. [{a.title}]({a.url})")
                    else:
                        st.markdown(f"{idx}. {a.title}")
                    if hasattr(a, "source") and a.source:
                        st.caption(f"📰 {a.source}")
    
    # Show summaries if requested
    if hasattr(st.session_state, 'show_summaries') and st.session_state.show_summaries:
        st.markdown("---")
        st.subheader("📝 Article Summaries")
        
        if not st.session_state.news_basket:
            st.info("No articles in basket. Add articles using the 📦 button.")
        else:
            # Get full article objects for summarization
            basket_articles = []
            for basket_item in st.session_state.news_basket:
                article = all_articles.get(basket_item['id'])
                if article:
                    basket_articles.append(article)
            
            if basket_articles:
                for i, article in enumerate(basket_articles, 1):
                    with st.expander(f"📰 {i}. {article.title}", expanded=True):
                        col_summary, col_actions = st.columns([3, 1])
                        
                        with col_summary:
                            summary = simple_summarize(article)
                            st.markdown(f"**Summary:** {summary}")
                            
                            if article.url:
                                st.markdown(f"**Link:** [{article.title}]({article.url})")
                            
                            if hasattr(article, 'source') and article.source:
                                st.caption(f"📰 Source: {article.source}")
                        
                        with col_actions:
                            if st.button("🗑️ Remove", key=f"remove_summary_{article.id}"):
                                st.session_state.news_basket = [item for item in st.session_state.news_basket if item['id'] != article.id]
                                st.rerun()
                
                # Clear summaries button
                if st.button("🗑️ Clear All Summaries"):
                    st.session_state.show_summaries = False
                    st.rerun()
            else:
                st.warning("Could not load article details for summarization.")


if __name__ == "__main__":
    main()


