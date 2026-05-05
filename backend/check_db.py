import sqlite3
conn = sqlite3.connect('app/data/drugbank.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('Tables:', c.fetchall())
try:
    c.execute('SELECT COUNT(*) FROM drugbank_synonyms')
    print('Synonyms:', c.fetchone())
except Exception as e:
    print('No synonyms table:', e)
conn.close()
