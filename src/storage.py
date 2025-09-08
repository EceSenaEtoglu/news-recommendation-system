import sqlite3
import json
from datetime import datetime
from typing import List, Optional
from .models import Article, Source, ContentType, TargetAudience
from datetime import datetime, timedelta
from math import exp
from.retrieval import RAGConfig

# TODO match with the actual api call
# TODO for simplicity did not use ORM, use ORM in the future
class ArticleDB:
    """Simple SQLite storage for articles"""
    
    def __init__(self, db_path: str = "db/articles.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                description TEXT,
                url TEXT NOT NULL,
                author TEXT,
                url_to_image TEXT,
                
                -- Source info
                source_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_credibility REAL,
                
                -- Bias detection
                content_type TEXT,
                target_audience TEXT,
                urgency_score REAL,
                
                -- Timestamps
                published_at TEXT,
                created_at TEXT,
                
                -- Embeddings (stored as JSON for now)
                embedding TEXT,
                entities TEXT,
                topics TEXT
            )
        """)
        
        # graph tables
        conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            type TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS article_entities (
            article_id TEXT NOT NULL,
            entity_id INTEGER NOT NULL,
            mentions INTEGER DEFAULT 1,
            PRIMARY KEY (article_id, entity_id),
            FOREIGN KEY(entity_id) REFERENCES entities(id)
        )
        """)
        
        # --- User events: read/save/skip with optional dwell time
        conn.execute("""
        CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        article_id TEXT NOT NULL,
        event_type TEXT NOT NULL CHECK(event_type IN ('read','save','skip')),
        dwell_ms INTEGER,
        ts TEXT NOT NULL
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events(user_id, ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_article ON events(article_id)")

        # --- Learned preferences (keyed by user + preference key)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS user_prefs(
        user_id TEXT NOT NULL,
        key TEXT NOT NULL,           -- e.g. 'ent:nvidia'
        weight REAL NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(user_id, key)
        )
        """)

        # Create index for faster searches
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_article_entities_entity ON article_entities(entity_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_article_entities_article ON article_entities(article_id)")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON articles(published_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON articles(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_content_type ON articles(content_type)")
        
        conn.commit()
        conn.close()
    
    def save_article(self, article: Article) -> bool:
        """Save an article to the database"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO articles (
                    id, title, content, description, url, author, url_to_image,
                    source_id, source_name, source_credibility,
                    content_type, target_audience, urgency_score,
                    published_at, created_at,
                    embedding, entities, topics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article.id,
                article.title,
                article.content,
                article.description,
                article.url,
                article.author,
                article.url_to_image,
                article.source.id,
                article.source.name,
                article.source.credibility_score,
                article.content_type.value,
                article.target_audience.value,
                article.urgency_score,
                article.published_at.isoformat(),
                article.created_at.isoformat(),
                json.dumps(article.embedding) if article.embedding else None,
                json.dumps(article.entities),
                json.dumps(article.topics)
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving article: {e}")
            return False
        finally:
            conn.close()
    
    def save_articles(self, articles: List[Article]) -> int:
        """Save multiple articles, return count saved"""
        saved = 0
        for article in articles:
            if self.save_article(article):
                saved += 1
        return saved
    
    def get_article_by_id(self, article_id: str) -> Optional[Article]:
        """Get a single article by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        
        try:
            cursor = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_article(row)
            return None
        finally:
            conn.close()
    
    def search_articles(self, 
                       query: str = "", 
                       source_ids: List[str] = None,
                       content_types: List[ContentType] = None,
                       limit: int = 10,
                       hours_back: int = 24) -> List[Article]:
        """Simple text search in articles"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Build SQL query
            # TODO what is 1 = 1?
            sql = "SELECT * FROM articles WHERE 1=1"
            params = []
            
            # Text search in title and content
            if query.strip():
                sql += " AND (title LIKE ? OR content LIKE ?)"
                query_param = f"%{query}%"
                params.extend([query_param, query_param])
            
            # Filter by sources
            if source_ids:
                placeholders = ",".join(["?" for _ in source_ids])
                sql += f" AND source_id IN ({placeholders})"
                params.extend(source_ids)
            
            # Filter by content types
            if content_types:
                placeholders = ",".join(["?" for _ in content_types])
                sql += f" AND content_type IN ({placeholders})"
                params.extend([ct.value for ct in content_types])
            
            # Recent articles only
            if hours_back > 0:
                from datetime import timedelta
                cutoff = (datetime.utcnow() - timedelta(hours=hours_back))
                sql += " AND published_at >= ?"
                params.append(cutoff.isoformat())
            
            # Order by recency and limit
            sql += " ORDER BY published_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            
            return [self._row_to_article(row) for row in rows]
            
        finally:
            conn.close()
    
    def get_recent_articles(self, limit: int = 50, hours_back: int = 24) -> List[Article]:
        """Get recent articles"""
        return self.search_articles(limit=limit, hours_back=hours_back)
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM articles")
            total_articles = cursor.fetchone()[0]
            
            cursor = conn.execute("""
                SELECT content_type, COUNT(*) 
                FROM articles 
                GROUP BY content_type
            """)
            content_type_counts = dict(cursor.fetchall())
            
            cursor = conn.execute("""
                SELECT source_name, COUNT(*) 
                FROM articles 
                GROUP BY source_name 
                ORDER BY COUNT(*) DESC 
                LIMIT 5
            """)
            top_sources = dict(cursor.fetchall())
            
            return {
                "total_articles": total_articles,
                "content_types": content_type_counts,
                "top_sources": top_sources
            }
        finally:
            conn.close()
    
    def _row_to_article(self, row) -> Article:
        """Convert SQLite row to Article object"""
        # Recreate source
        source = Source(
            id=row["source_id"],
            name=row["source_name"],
            url="",  # Not stored separately
            category="general",
            credibility_score=row["source_credibility"] or 0.5
        )
        
        # Parse timestamps
        published_at = datetime.fromisoformat(row["published_at"])
        created_at = datetime.fromisoformat(row["created_at"])
        
        # Parse JSON fields
        embedding = json.loads(row["embedding"]) if row["embedding"] else None
        entities = json.loads(row["entities"]) if row["entities"] else []
        topics = json.loads(row["topics"]) if row["topics"] else []
        
        return Article(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            description=row["description"],
            url=row["url"],
            source=source,
            published_at=published_at,
            author=row["author"],
            url_to_image=row["url_to_image"],
            content_type=ContentType(row["content_type"]),
            target_audience=TargetAudience(row["target_audience"]),
            urgency_score=row["urgency_score"],
            embedding=embedding,
            entities=entities,
            topics=topics,
            created_at=created_at
        )
        
    # Batch loading increases efficiency
    def get_articles_by_ids(self, article_ids: List[str]) -> List[Article]:
        """Get multiple articles by their IDs in a single query."""
        if not article_ids:
            return []
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Create placeholders for IN clause
            placeholders = ",".join(["?" for _ in article_ids])
            sql = f"SELECT * FROM articles WHERE id IN ({placeholders})"
            
            cursor = conn.execute(sql, article_ids)
            rows = cursor.fetchall()
            
            return [self._row_to_article(row) for row in rows]
            
        finally:
            conn.close()
            
    # graph helpers
    def upsert_article_entities(self, article_id: str, named_entities: list[tuple[str,str,int]]) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            for name, etype, mentions in named_entities:
                conn.execute("INSERT OR IGNORE INTO entities(name, type) VALUES(?,?)", (name, etype))
                eid = conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
                if not eid: 
                    continue
                conn.execute("""
                INSERT INTO article_entities(article_id, entity_id, mentions)
                VALUES(?,?,?)
                ON CONFLICT(article_id, entity_id) DO UPDATE SET
                    mentions = article_entities.mentions + excluded.mentions
                """, (article_id, eid[0], max(1, int(mentions))))
            conn.commit()
        finally:
            conn.close()

    def get_articles_by_entities(self, entities: list[str], limit: int = 50, hours_back: int = 24*7) -> list[Article]:
        if not entities: return []
        conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" for _ in entities)
            e_rows = conn.execute(f"SELECT id FROM entities WHERE name IN ({placeholders})", entities).fetchall()
            if not e_rows: return []
            eids = [r[0] for r in e_rows]
            cutoff = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
            placeholders2 = ",".join("?" for _ in eids)
            rows = conn.execute(f"""
                SELECT a.*
                FROM article_entities ae
                JOIN articles a ON a.id = ae.article_id
                WHERE ae.entity_id IN ({placeholders2}) AND a.published_at >= ?
                GROUP BY ae.article_id
                ORDER BY SUM(ae.mentions) DESC, a.published_at DESC
                LIMIT ?
            """, eids + [cutoff, limit]).fetchall()
            return [self._row_to_article(r) for r in rows]
        finally:
            conn.close()

    def get_comention_counts(self, seed_entities: list[str], limit: int = 50) -> dict[str,int]:
        if not seed_entities: return {}
        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ",".join("?" for _ in seed_entities)
            seed_ids = [r[0] for r in conn.execute(f"SELECT id FROM entities WHERE name IN ({placeholders})", seed_entities)]
            if not seed_ids: return {}
            placeholders2 = ",".join("?" for _ in seed_ids)
            art_ids = [r[0] for r in conn.execute(
                f"SELECT DISTINCT article_id FROM article_entities WHERE entity_id IN ({placeholders2})", seed_ids
            )]
            if not art_ids: return {}
            placeholders3 = ",".join("?" for _ in art_ids)
            rows = conn.execute(f"""
                SELECT e.name, SUM(ae.mentions) AS c
                FROM article_entities ae
                JOIN entities e ON e.id = ae.entity_id
                WHERE ae.article_id IN ({placeholders3})
                GROUP BY ae.entity_id
                ORDER BY c DESC
                LIMIT ?
            """, art_ids + [limit]).fetchall()
            return {name: int(c) for (name, c) in rows if name not in seed_entities}
        finally:
            conn.close()
            
    def log_event(self, user_id: str, article_id: str, event_type: str, dwell_ms: int | None = None, ts: datetime | None = None):
        """Insert a user interaction event."""
        ts = ts or datetime.utcnow()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO events(user_id, article_id, event_type, dwell_ms, ts) VALUES(?,?,?,?,?)",
                (user_id, article_id, event_type, dwell_ms, ts.isoformat())
            )
            conn.commit()
        finally:
            conn.close()

    def get_recent_events(self, user_id: str, days: int = 14, limit: int = 5000) -> list[sqlite3.Row]:
        """Fetch recent events for a user (for preference updates)."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM events WHERE user_id = ? AND ts >= ? ORDER BY ts DESC LIMIT ?",
                (user_id, since, limit)
            ).fetchall()
            return rows
        finally:
            conn.close()

    def get_user_prefs(self, user_id: str) -> dict[str, float]:
        """Return {pref_key: weight} for a user."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("SELECT key, weight FROM user_prefs WHERE user_id = ?", (user_id,)).fetchall()
            return {k: float(w) for (k, w) in rows}
        finally:
            conn.close()

    def upsert_user_pref_weights(
    self, user_id: str, deltas: dict[str, float], clip: tuple[float,float] = (-2.0, 2.0),clip_map: dict[str, tuple[float,float]] | None = None):
        """
        Incrementally update preference weights with clipping.
        - clip: default clip if prefix not found in clip_map
        - clip_map: optional per-prefix clip, e.g. {'ent': (-2, 2), 'topic': (-1.5, 1.5)}
        """
        if not deltas:
            return
        conn = sqlite3.connect(self.db_path)
        try:
            now = datetime.utcnow().isoformat()
            for key, delta in deltas.items():
                prefix = key.split(":", 1)[0] if ":" in key else ""
                lo, hi = (clip_map.get(prefix) if clip_map and prefix in clip_map else clip)

                row = conn.execute(
                    "SELECT weight FROM user_prefs WHERE user_id=? AND key=?",
                    (user_id, key)
                ).fetchone()
                if row:
                    new_w = max(lo, min(hi, float(row[0]) + float(delta)))
                    conn.execute(
                        "UPDATE user_prefs SET weight=?, updated_at=? WHERE user_id=? AND key=?",
                        (new_w, now, user_id, key)
                    )
                else:
                    new_w = max(lo, min(hi, float(delta)))
                    conn.execute(
                        "INSERT INTO user_prefs(user_id, key, weight, updated_at) VALUES(?,?,?,?)",
                        (user_id, key, new_w, now)
                    )
            conn.commit()
        finally:
            conn.close()


    def recompute_prefs_from_events(self, user_id: str, days: int = 14):
        """
        Rebuild per-entity and per-topic preference weights from recent user events.
        Uses separate learning rates and clip ranges for entities vs topics.

        SIGNALS & SCORING
        -----------------
        delta = base_weight(event) * exp(-hours_since_pub / (4*24)) * dwell_factor
        - base_weight: save=+1.0, read=+0.5, skip=-0.3
        - freshness: 4-day time constant (0h=1.0, 1d≈0.78, 4d≈0.37, 8d≈0.14)
        - dwell_factor: 1.0 if dwell_ms<=0 else min(1.5, 0.5 + dwell_ms/30000)

        LEARNING
        --------
        - Entities: LR=1.0, clip [-2.0, 2.0]
        - Topics:   LR=0.5, clip [-1.5, 1.5]
        - Per-article: normalize → dedup (order-preserving) → cap to 10.
        """
        events = self.get_recent_events(user_id, days=days)
        if not events:
            return

        article_ids = list({row["article_id"] for row in events})
        articles_by_id = {a.id: a for a in self.get_articles_by_ids(article_ids)}

        base = {"save": 1.0, "read": 0.5, "skip": -0.3}
   
        deltas: dict[str, float] = {}
        now = datetime.utcnow()

        for ev in events:
            a = articles_by_id.get(ev["article_id"])
            if not a:
                continue

            pub = getattr(a, "published_at", None)
            if pub and isinstance(pub, datetime):
                hours = max(1.0, (now - pub).total_seconds() / 3600.0)
            else:
                hours = 24.0  # safe fallback

            fresh = exp(-hours / (RAGConfig.freshness_decay_exp_constant * 24.0))

            dwell_ms = ev["dwell_ms"] or 0
            dwell_factor = 1.0 if dwell_ms <= 0 else min(1.5, 0.5 + (dwell_ms / 30000.0))

            delta = base.get(ev["event_type"], 0.0) * fresh * dwell_factor
            if delta == 0.0:
                continue

            # Normalize first, then dedup, then cap.
            ents_raw   = (getattr(a, "entities", None) or [])
            topics_raw = (getattr(a, "topics", None) or [])
            ents_norm   = [str(e or "").strip().lower() for e in ents_raw if e is not None]
            topics_norm = [str(t or "").strip().lower() for t in topics_raw if t]
            ents   = list(dict.fromkeys(ents_norm))[:RAGConfig.max_entities_per_article]
            topics = list(dict.fromkeys(topics_norm))[:RAGConfig.max_topics_per_article]

            for e in ents:
                key = f"ent:{e}"
                deltas[key] = deltas.get(key, 0.0) + RAGConfig.entity_lr* delta

            for t in topics:
                key = f"topic:{t}"
                deltas[key] = deltas.get(key, 0.0) + RAGConfig.topic_lr* delta

        # Fresh recompute
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM user_prefs WHERE user_id = ?", (user_id,))
            conn.commit()
        finally:
            conn.close()

        # Separate clips
        self.upsert_user_pref_weights(
            user_id,
            deltas,
            clip=(-2.0, 2.0),
            clip_map={'ent': (-2.0, 2.0), 'topic': (-1.5, 1.5)}
        )


        