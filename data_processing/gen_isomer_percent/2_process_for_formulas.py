import sqlite3


def add_column_if_missing(conn, table_name, column_def):
    cur = conn.cursor()
    col_name = column_def.split()[0]
    cur.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in cur.fetchall()]
    if col_name not in columns:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_def}")
    conn.commit()


def main(db_path="your_database.sqlite"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Insert placeholder into formulas
    cur.execute("""
        INSERT OR IGNORE INTO formulas (id, value)
        VALUES (-1, 'placeholder')
    """)

    # 2. Count occurrences of each formula_id
    cur.execute("""
        CREATE TEMP TABLE formula_counts AS
        SELECT formula_id, COUNT(*) as cnt
        FROM main
        GROUP BY formula_id
    """)

    # 3. Replace formula_id in main with -1 where count == 1
    cur.execute("""
        UPDATE main
        SET formula_id = -1
        WHERE formula_id IN (
            SELECT formula_id FROM formula_counts WHERE cnt = 1
        )
    """)

    # 4. Delete formulas with count == 1
    cur.execute("""
        DELETE FROM formulas
        WHERE id IN (
            SELECT formula_id FROM formula_counts WHERE cnt = 1
        )
    """)

    # 5. Add new column for reindexing
    #    If it already exists from a previous run, ignore error
    add_column_if_missing(conn, "formulas", "new_id INTEGER")

    # Assign new sequential IDs starting at 0, excluding placeholder (-1)
    cur.execute("""
        WITH ordered AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY id) - 1 AS seq
            FROM formulas
            WHERE id != -1
        )
        UPDATE formulas
        SET new_id = (
            SELECT seq FROM ordered WHERE ordered.id = formulas.id
        )
        WHERE id != -1
    """)

    # Keep placeholder at -1
    cur.execute("UPDATE formulas SET new_id = -1 WHERE id = -1")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main("organized_data_train.sqlite3")