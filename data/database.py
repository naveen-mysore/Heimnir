import sqlite3
import pandas as pd


class DatabaseHandler:
    """
    Handler for the SQLite database to store data into SQLite.
    This class preprocesses the data for benchmarking and stores it in the database.
    """

    def __init__(self, path_to_db, path_to_schema, name):
        """
        Initializes the SQLite database and creates tables if they do not exist.
        """
        self.name = name
        self.db_path = path_to_db
        self.schema_path = path_to_schema
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def get_connection(self):
        return self.conn

    def get_cursor(self):
        return self.cursor

    def run_a_sql_query(self, query, params):
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        return rows

    def commit(self):
        self.conn.commit()

    def close_connection(self):
        self.conn.close()

    def clear_all_and_create_tables(self):
        """
        Create all necessary tables based on the schema.
        """
        with open(self.schema_path, 'r') as sql_file:
            self.conn.executescript(sql_file.read())
        self.conn.commit()

    def load_csv_to_sqlite(self, csv_path, table_name):
        """
        Loads data from a CSV file into the specified SQLite table.
        Args:
            csv_path (str): Path to the CSV file.
            table_name (str): Name of the table in the database.
        """
        try:
            # Load the CSV file with UTF-8 encoding
            data = pd.read_csv(csv_path, encoding='utf-8')
            # Insert the data into the SQLite database
            data.to_sql(table_name, self.conn, if_exists='append', index=False)
            self.commit()

        except Exception as e:
            print(f"Error loading CSV data into SQLite: {e}")

    def fetch_table_as_dataframe(self, table_name):
        """
        Fetches all data from a given table and returns it as a pandas DataFrame.
        Args:
            table_name (str): The name of the table to fetch data from.
        Returns:
            pd.DataFrame: DataFrame containing all the data from the specified table.
        """
        try:
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            return pd.DataFrame()  # Return an empty DataFrame in case of error

    def close(self):
        """Close the database connection."""
        self.conn.close()
