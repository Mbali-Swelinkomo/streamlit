import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('mydatabase.db')

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Execute a simple query to fetch data from the table
cursor.execute('SELECT * FROM mytable')

# Fetch all rows from the result set
rows = cursor.fetchall()

# Print the rows
for row in rows:
    print(row)

# Close the cursor and connection
cursor.close()
conn.close()
