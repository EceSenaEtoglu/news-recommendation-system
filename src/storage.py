import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Optional
from .data_models import Article, Source, ContentType

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
                
                -- Labels
                content_type TEXT,
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
                    content_type, urgency_score,
                    published_at, created_at,
                    embedding, entities, topics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            # trick, get all articles regardless of the filters
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
                from datetime import timezone
                cutoff = (datetime.now(timezone.utc)- timedelta(hours=hours_back))
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
            cutoff = (datetime.now(datetime.timezone.utc)- timedelta(hours=hours_back)).isoformat()
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
            


        