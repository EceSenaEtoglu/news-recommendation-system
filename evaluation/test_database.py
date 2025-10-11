"""
Test Database Setup for SPICED Evaluation
Creates a test database populated with SPICED articles for proper evaluation.
"""

import os
import sys
import sqlite3
import json
from datetime import datetime, timezone
from typing import List, Dict
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

sys.path.insert(0, os.path.join(project_root, 'src'))
from data_models import Article, Source, ContentType, SourceCategory

class SPICEDTestDatabase:
    """Creates and manages a test database with SPICED articles."""
    
    def __init__(self, db_path: str = "evaluation/test_db/test_spiced.db"):
        self.db_path = db_path
        self.spiced_data = None
        
    def load_spiced_data(self) -> bool:
        """Load SPICED dataset."""
        try:
            self.spiced_data = pd.read_csv('spiced_data/spiced.csv')
            print(f"Loaded SPICED dataset: {len(self.spiced_data)} pairs")
            return True
        except Exception as e:
            print(f"Failed to load SPICED dataset: {e}")
            return False
    
    def create_test_database(self) -> bool:
        """Create test database with SPICED articles."""
        if self.spiced_data is None:
            if not self.load_spiced_data():
                return False
        
        try:
            # Remove existing test database
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            
            # Create new database
            conn = sqlite3.connect(self.db_path)
            
            # Create tables (matching ArticleDB schema)
            conn.execute("""
                CREATE TABLE articles (
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
                    
                    -- Provenance (added for agentic journalist reports)
                    provenance_source TEXT,
                    decision_type TEXT,
                    evidence_urls TEXT,
                    
                    -- Timestamps
                    published_at TEXT,
                    created_at TEXT,
                    
                    -- Embedding
                    embedding TEXT,
                    
                    -- Entities and topics
                    entities TEXT,
                    topics TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE sources (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    url TEXT,
                    category TEXT,
                    credibility_score REAL DEFAULT 0.7,
                    created_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT,
                    created_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE article_entities (
                    article_id TEXT,
                    entity_id INTEGER,
                    PRIMARY KEY (article_id, entity_id),
                    FOREIGN KEY (article_id) REFERENCES articles(id),
                    FOREIGN KEY (entity_id) REFERENCES entities(id)
                )
            """)
            
            # Create default source
            default_source = Source(
                id="spiced_source",
                name="SPICED Test Source",
                url="https://spiced-test.com",
                category=SourceCategory.GENERAL,
                credibility_score=0.8
            )
            
            conn.execute("""
                INSERT INTO sources (id, name, url, category, credibility_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                default_source.id,
                default_source.name,
                default_source.url,
                default_source.category.value,
                default_source.credibility_score,
                datetime.now(timezone.utc).isoformat()
            ))
            
            # Insert SPICED articles
            article_count = 0
            current_time = datetime.now(timezone.utc).isoformat()
            
            for idx, row in self.spiced_data.iterrows():
                # Create article from text_1
                article_id_1 = f"spiced_{idx}_1"
                article_1 = Article(
                    id=article_id_1,
                    title=f"SPICED Article {idx} - Part 1",
                    description=row['text_1'][:200] + "...",
                    content=row['text_1'],
                    url=row['URL_1'],
                    source=default_source,
                    published_at=datetime.now(timezone.utc),
                    content_type=ContentType.FACTUAL,
                    topics=[row['Type']],
                    urgency_score=0.5
                )
                
                conn.execute("""
                    INSERT INTO articles (
                        id, title, content, description, url, author, url_to_image,
                        source_id, source_name, source_credibility,
                        content_type, urgency_score,
                        provenance_source, decision_type, evidence_urls,
                        published_at, created_at,
                        embedding, entities, topics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article_1.id, article_1.title, article_1.content, article_1.description,
                    article_1.url, None, None,
                    article_1.source.id, article_1.source.name, article_1.source.credibility_score,
                    article_1.content_type.value, article_1.urgency_score,
                    None, None, json.dumps([]),
                    article_1.published_at.isoformat(), current_time,
                    None, json.dumps([]), json.dumps(article_1.topics)
                ))
                
                # Create article from text_2
                article_id_2 = f"spiced_{idx}_2"
                article_2 = Article(
                    id=article_id_2,
                    title=f"SPICED Article {idx} - Part 2",
                    description=row['text_2'][:200] + "...",
                    content=row['text_2'],
                    url=row['URL_2'],
                    source=default_source,
                    published_at=datetime.now(timezone.utc),
                    content_type=ContentType.FACTUAL,
                    topics=[row['Type']],
                    urgency_score=0.5
                )
                
                conn.execute("""
                    INSERT INTO articles (
                        id, title, content, description, url, author, url_to_image,
                        source_id, source_name, source_credibility,
                        content_type, urgency_score,
                        provenance_source, decision_type, evidence_urls,
                        published_at, created_at,
                        embedding, entities, topics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article_2.id, article_2.title, article_2.content, article_2.description,
                    article_2.url, None, None,
                    article_2.source.id, article_2.source.name, article_2.source.credibility_score,
                    article_2.content_type.value, article_2.urgency_score,
                    None, None, json.dumps([]),
                    article_2.published_at.isoformat(), current_time,
                    None, json.dumps([]), json.dumps(article_2.topics)
                ))
                
                article_count += 2
            
            conn.commit()
            conn.close()
            
            print(f"Created test database with {article_count} SPICED articles")
            print(f"Database path: {self.db_path}")
            return True
            
        except Exception as e:
            print(f"Failed to create test database: {e}")
            return False
    
    def get_database_path(self) -> str:
        """Get the test database path."""
        return self.db_path
    
    def verify_database(self) -> bool:
        """Verify the test database was created correctly."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check article count
            article_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            print(f"Database contains {article_count} articles")
            
            # Check source count
            source_count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            print(f"Database contains {source_count} sources")
            
            # Sample articles
            sample_articles = conn.execute("""
                SELECT id, title, topics FROM articles LIMIT 5
            """).fetchall()
            
            print("Sample articles:")
            for article in sample_articles:
                print(f"  {article[0]}: {article[1]} - {article[2]}")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"Failed to verify database: {e}")
            return False

def main():
    """Create test database for SPICED evaluation."""
    print("Creating SPICED Test Database")
    print("=" * 40)
    
    test_db = SPICEDTestDatabase()
    
    if test_db.create_test_database():
        print("Test database created successfully")
        test_db.verify_database()
    else:
        print("Failed to create test database")

if __name__ == "__main__":
    main()
