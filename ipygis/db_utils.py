import os


def get_connection_url(user=os.environ.get('PGUSER', 'postgres'),
                       dbname=os.environ.get('PGDATABASE', 'postgres'),
                       port=int(os.environ.get('PGPORT', '5432')), host=os.environ.get('PGHOST', 'localhost'),
                       password=os.environ.get('PGPASSWORD', '')) -> str:
    """
    Creates connection string for PostgreSQL connections
    """
    c_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return c_url
