---
title: MBAI 417
separator: <!--s-->
verticalSeparator: <!--v-->
theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Data Intensive Systems
  ## L.02 | Introduction to Databases

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 10%">

  <iframe src="https://lottie.host/embed/216f7dd1-8085-4fd6-8511-8538a27cfb4a/PghzHsvgN5.lottie" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Welcome to Data Intensive Systems.
  ## Please log in and enter the code on the whiteboard.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 50%; padding-top: 0%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

## Announcements

89% of you have submitted the first assignment!  🎉
  - Please make sure to submit it by **Thursday 11:59 PM**.
  - If you're still having trouble, use the [Codespaces setup](https://canvas.northwestern.edu/courses/231041/discussion_topics/1641663)


<!--s-->

## De-Identified Responses to Coverage Poll

```python
'NO',
'I THINK THERE IS PLENTY ON THE SYLLABUS',
'NO THANKS',
'N/A - LOOK FORWARD TO THE CURRENT PLAN',
'VECTOR DATABASE, MCP', #We'll cover both of these in Modern NLP Applications.
'N/A. LOOKS GOOD!',
'I WOULD SUGGEST TO TAKE IT STEP BY STEP AND GRADUALLY RAMP UP', # Will do!
'NONE',
"YOU'RE COVERING EVERYTHING!",
'STRUCTURING EFFICIENT DATABASES / LEVERAGING LARGE AMOUNTS OF DATA EFFICIENTLY', # OLAP & Distributed Preprocessing will cover this.
'NA',
'PLEASE COVER IT ALL! THERE ARE DIFFERENT LEVELS OF EXPERIENCE IN THIS CLASS, AND WHILE IT MAY BE REDUNDANT FOR SOME, A GOOD NUMBER OF US WOULD ALSO BENEFIT FROM REFRESHES AND A SLIGHTLY SLOWER PACE', # Will do!
'HOW TO EFFICIENTLY STORE IMAGES, VIDEOS? ', # Added to today's lecture!
"I DON'T KNOW WHAT I DON'T KNOW :)", # Fair enough. :)
'ONLINE LEARNING SOURCES - BLOGS, PODCASTS ETC', # I'll cite sources as we go.
'ANYTHING!',
'NA',
'MORE RAG!', # Two lectures will cover related topics to RAG. 
'NO, ALL LOOKS GOOD!',
'LDAP AND USER PERMISSIONS, DATA COMPUTE ON THE EDGE', # User permissions and basic security will be covered on Thursday, I don't have any plans for edge computing though.
'NONE, LOOKING FORWARD TO IT!',
'TEST', # Did it work ?
'LOOKS GREAT TO ME!',
'NOPE!',
'VERY EXCITED FOR THE EMPHASIS ON GUI! NOTHING OTHERWISE', # Love a good GUI!
'NO NOTES - LOOKS GOOD!',
'NOPE',
'FEEDING DATA INTO AI, DATA PREPARATION', # 2-3 lectures will cover this!
'AUTHENTICATION DATABASES', # Will be covered Thursday. 
'NOT THAT I CAN THINK OF',
'NOPE!',
'WOULD LIKE TO LEARN ABOUT KUBERNETES AND DATACENTER OPS' # We do learn about containerization, and I'll make sure to give a couple of examples on orchestration.
```

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with navigating database options & basic SQL?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>
<!--s-->

## Agenda

### Databases

<div style = 'margin-left: 5%'>

#### What is a database?
#### Why do we need databases?
#### What makes a 'good' database?

</div>

### Database Landscape

<div style = 'margin-left: 5%'>

#### What are the different types of databases?

<!--s-->

<div class="header-slide">

# Databases

</div>

<!--s-->

## Databases | What is a database?

A database is a collection of data that is organized in a way that allows for efficient retrieval and manipulation. At it's core, a database is simply a structured way to **store** and **manage** data on a computer. 

A database can be as simple as a text file or a spreadsheet, or as complex as a distributed system that spans multiple servers and locations.

<div style="text-align: center;">
    <img src="https://guide-images.cdn.ifixit.com/igi/BCU4AgbFicGvFcZA.large" width="40%" style="border-radius: 10px;" />
    <p style="text-align: center; font-size: 0.6em; color: grey;">iFixit</p>
</div>

<!--s-->

## Databases | What makes a **good** database?

- ### Model
- ### Integrity
- ### Access
- ### Security
- ### Scalability

<!--s-->

## Databases | Model

Databases start with a data **model**. A data model is a conceptual representation of the data and its relationships, and it serves as a blueprint for how the data will be stored, organized, and accessed. Different data models are suited for different types of data and use cases.

<div style='text-align: center;'>
   <img src='https://www.gooddata.com/img/blog/_2000xauto/pdm-for-e-commerce.png.webp' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0;'>Gooddata 2025</p>
</div>

<!--s-->

## Databases | Integrity

Data integrity refers to the accuracy and consistency of data over its entire lifecycle. A good database should enforce data integrity constraints to ensure that the data is **valid** and **reliable**.

<div class = "col-wrapper" style = "font-size: 0.9em;">
<div class="c1" style = "width: 50%; margin-right: 1em;">

### Constraints
Constraints are rules that define the valid values and relationships for data in a database. They help maintain data integrity by preventing invalid data from being entered into the database.

### Transactions
A transaction is a sequence of operations that are treated as a single unit of work. Transactions ensure that either all operations are completed successfully or none are, maintaining data integrity in the event of errors or failures.

</div>
<div class="c2" style = "width: 50%">

### Normalization
Normalization is the process of organizing data in a database to minimize redundancy and dependency. It helps maintain data integrity by ensuring that data is stored in a logical and efficient manner.

</div>
</div>

<!--s-->

## Databases | Access

Data access refers to the methods and mechanisms used to retrieve and manipulate data in a database. A good database should provide efficient and flexible data access methods to meet the needs of different applications and users.

Some methods of access include SQL, APIs, GraphQL, and NoSQL queries.

We'll talk more about access in L.03.

<!--s-->

## Databases | Security

Data security refers to the measures taken to protect data from unauthorized access, corruption, or loss. A good database should implement robust security measures to safeguard sensitive data and ensure compliance with regulations.

<div class = "col-wrapper" style = "font-size: 0.9em;">
<div class="c1" style = "width: 50%; margin-right: 1em;">

### Authentication
Authentication is the process of verifying the identity of users or applications accessing the database. Support strong authentication mechanisms to ensure that only authorized users can access the data.

### Permissions
Permissions are rules that define what actions users or applications can perform on the data. Provide fine-grained access control to ensure that users can only access the data they are authorized to see.

</div>
<div class="c2" style = "width: 50%">

### Encryption
Encryption is the process of converting data into a secure format that can only be read by authorized users. Use encryption to protect sensitive data both at rest and in transit.

</div>
</div>

<!--s-->

## Databases | Scalability

Data scalability refers to the ability of a database to handle increasing amounts of data and users without sacrificing performance. A good database should be designed to scale horizontally or vertically as needed.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Vertically
Adding more resources to a single server.

### Horizontally
Adding more servers to distribute the load.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center; '>
   <img src='https://images.ctfassets.net/00voh0j35590/6wtOJjoIPbeqctg7dzjGS4/ca386d6416546a8ba6957e7b6407c5e4/vertical-versus-horizontal-scaling-compared-diagram.jfif' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Cockroach Labs</p>
</div>

</div>
</div>




<!--s-->

<div class="header-slide">

# Modern Databases

</div>

<!--s-->

# Database Landscape

<div style='font-size: 0.65em'>

| Type | Description | Use Cases | Examples |
| --- | --- | --- | --- |
| Relational | Structured data with relationships | Transactional applications | MySQL, PostgreSQL |
| Key-Value | Simple key-value pairs | Caching, session management | Redis, DynamoDB |
| Document | Semi-structured data in documents | Content management, e-commerce | MongoDB, Couchbase |
| Graph | Data with complex relationships | Social networks, recommendation systems | Neo4j |
| Wide-Column | Large-scale data with flexible schema | Big data analytics | Cassandra, BigTable |
| In-Memory | Fast access to data in memory | Real-time analytics, caching | Redis, Memcached |
| Time-Series | Data with time-based relationships | IoT, financial data | InfluxDB, TimescaleDB |
| Object-Oriented | Data as objects with methods | CAD, multimedia applications | db4o, ObjectDB |
| Spatial | Geospatial data and queries | GIS, location-based services | PostGIS (PostgreSQL) |
| Blob Datastore | Unstructured data storage | Media files, backups | Amazon S3, Google Cloud Storage |
| Ledger | Immutable, tamper-proof data | Financial transactions, supply chain | Amazon Quantum Ledger Database |
| Hierarchical | Tree-like data structure | File systems, XML data | IBM IMS, Windows Registry |
| Vector | High-dimensional data for ML | Recommendation systems, image search | Pinecone, pgvector (PostgreSQL) |
| Embedded | Lightweight databases for applications | Mobile apps, IoT devices | SQLite, Realm |

</div>

<!--s-->

<div class = "header-slide">

# Relational Databases

</div>

<!--s-->

## Relational Databases

<img src="https://planetscale.com/assets/blog/content/schema-design-101-relational-databases/a2906fd68b050d7f9e0714c7d566990efd645005-1953x1576.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Ramos 2022</p>

<!--s-->

## Relational Databases

<img src="https://planetscale.com/assets/blog/content/schema-design-101-relational-databases/db72cc3ac506bec544588454972113c4dc3abe50-1953x1576.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Ramos 2022</p>

<!--s-->

## Relational Databases

Relational databases are a type of database management system (DBMS) that store and manage data in a structured format using tables. Each table, or relation, consists of rows and columns, where rows represent individual records and columns represent the attributes of the data.

### Key Vocabulary:

- **Tables:** Organized into rows and columns, with each row being a unique data entry and each column representing a data attribute.
- **Relationships:** Tables are connected through keys, with primary keys uniquely identifying each record and foreign keys linking related records across tables.
- **SQL (Structured Query Language):** The standard language used to query and manipulate data within a relational database.
- **Normalization:** The process of organizing data to reduce redundancy and improve data integrity by dividing large tables into smaller, related tables.

<!--s-->

## Relational Databases

Relational databases are extremely common. Examples of relational databases include **MySQL** and **PostgreSQL**.

Relational Databases use SQL (Structured Query Language) to query and manipulate their contained data. SQL is a powerful language that allows for complex queries, joins, and aggregations.

<!--s-->

<div style="font-size: 0.8em;">

## SQL Query Cheat Sheet (Part 1)

### `CREATE TABLE`

  ```sql
  /* Create a table called table_name with column1, column2, and column3. */
  CREATE TABLE table_name (
    column1 INT PRIMARY KEY, /* Primary key is a unique identifier for each row. */
    column2 VARCHAR(100), /* VARCHAR is a variable-length string up to 100 characters. */
    column3 DATE /* DATE is a date type. */
  );
  ```

### `INSERT INTO`

  ```sql
  /* Insert values into column1, column2, and column3 in table_name. */
  INSERT INTO table_name (column1, column2, column3) VALUES (value1, value2, value3);
  ```

### `UPDATE`

  ```sql
  /* Update column1 in table_name to 'value' where column2 is equal to 'value'. */
  UPDATE table_name SET column1 = 'value' WHERE column2 = 'value';
  ```

### `DELETE`

  ```sql
  /* Delete from table_name where column1 is equal to 'value'. */
  DELETE FROM table_name WHERE column1 = 'value';
  ```

</div>

<!--s-->

<div style="font-size: 0.8em;">

## SQL Query Cheat Sheet (Part 2)

### `SELECT`
  
  ```sql
  /* Select column1 and column2 from table_name.*/
  SELECT column1, column2 FROM table_name;
  ```

### `WHERE`

  ```sql
  /* Select column1 and column2 from table_name where column1 is equal to 'value' and column2 is equal to 'value'. */
  SELECT column1, column2 FROM table_name WHERE column1 = 'value' AND column2 = 'value';
  ```

### `ORDER BY`

  ```sql
  /* Select column1 and column2 from table_name and order by column1 in descending order. */
  SELECT column1, column2 FROM table_name ORDER BY column1 DESC;
  ```

### `LIMIT`

  ```sql
  /* Select column1 and column2 from table_name and limit the results to 10. */
  SELECT column1, column2 FROM table_name LIMIT 10;
  ```

</div>

<!--s-->

<div style="font-size: 0.8em;">

## SQL Query Cheat Sheet (Part 3)

### `JOIN`

  ```sql
  /* Select column1 and column2 from table1 and table2 where column1 is equal to column2. */
  SELECT column1, column2 FROM table1 JOIN table2 ON table1.column1 = table2.column2;
  ```

### `GROUP BY`

  ```sql
  /* Select column1 and column2 from table_name and group by column1. */
  SELECT column1, column2 FROM table_name GROUP BY column1;
  ```

### `COUNT`

  ```sql
  /* Select the count of column1 from table_name. */
  SELECT COUNT(column1) FROM table_name;

  /* Group by column2 and select the count of column1 from table_name. */
  SELECT column2, COUNT(column1) FROM table_name GROUP BY column2;
  ```

### `SUM`

  ```sql
  /* Select the sum of column1 from table_name. */
  SELECT SUM(column1) FROM table_name;

  /* Group by column2 and select the sum of column1 from table_name. */
  SELECT column2, SUM(column1) FROM table_name GROUP BY column2;
  ```

</div>

<!--s-->

## SQL and Pandas (🔥)

```python
import pandas as pd
import sqlite3

# Create a connection to a SQLite database.
conn = sqlite3.connect('example.db')

# Load a DataFrame into the database.
df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']})
df.to_sql('table_name', conn, if_exists='replace')

# Query the database.
query = 'SELECT * FROM table_name'
df = pd.read_sql(query, conn)
```

<!--s-->

## Common Table Expression (CTE)

Common Table Expressions are temporary result sets that can be referenced within a <span class="code-span">SELECT</span>, <span class="code-span">INSERT</span>, <span class="code-span">UPDATE</span>, or <span class="code-span">DELETE</span> statement. They are defined using the <span class="code-span">WITH</span> keyword and can be used to simplify complex queries by breaking them down into smaller, more manageable parts. CTEs can also be recursive, allowing for hierarchical data retrieval.

Below is an example that first defines a CTE (which essentially creates a temporary table) and then uses it in a query.

```sql

WITH bounds AS (
  SELECT MIN(column1) AS min_value, MAX(column1) AS max_value
  FROM table_1_name
)

SELECT column1, column2
FROM table_2_name
WHERE column1 BETWEEN (SELECT min_value FROM bounds) AND (SELECT max_value FROM bounds)

```

<!--s-->

## SQL Aliases

Aliases are temporary names assigned to columns or tables in a SQL query. They are used to make the query more readable and concise, especially when dealing with complex queries or when joining multiple tables. Aliases are defined using the <span class="code-span">AS</span> keyword.

```sql
SELECT column1 AS alias1, column2 AS alias2
FROM table_name AS alias_table
```

<!--s-->


## L.02 | Q.01

Let's say you have a table called <span class="code-span">employees</span> with the following columns: <span class="code-span">id</span>, <span class="code-span">name</span>, <span class="code-span">department</span>, and <span class="code-span">salary</span>. 

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Write a SQL query to get the sum of the salaries of all employees in the <span class="code-span">Engineering</span> department.

| id | name | department | salary |
| --- | --- | --- | --- |
| 1 | Alice | Engineering | 50000 |
| 2 | Bob | Sales | 60000 |
| 3 | Charlie | Engineering | 70000 |
| 4 | David | Engineering | 80000 |

</div>
<div class="c2" style = "width: 50%">

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L.02 | Q.01" width = "100%" height = "100%"></iframe>

</div>
</div>

<!--s-->

<div class="header-slide">

# Key-Value Stores

</div>

<!--s-->

## Key-Value Stores

Key-value stores are a type of NoSQL database that store data as simple key-value pairs. Each key is unique, and it maps to a specific value. Key-value stores are designed for high performance and scalability, making them ideal for caching and session management. Key-value stores are often used in applications that require fast access to data, such as web applications, gaming, and real-time analytics.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Popular key-value stores include **Redis** and **Amazon DynamoDB**.

</div>
<div class="c2" style = "width: 50%">

<img src="https://www.scylladb.com/wp-content/uploads/Key-Value-Store-diagram-1-e1644958335886.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Scylla DB</p>

</div>
</div>

<!--s-->

<div class="header-slide">

# Document Stores

</div>

<!--s-->

## Document Stores

Document stores are another type of NoSQL database that store data in semi-structured documents, typically in JSON or BSON format. Each document can have a different structure, allowing for flexibility in data representation. Document stores are well-suited for content management systems, e-commerce applications, and any use case where data can vary in structure.

Some popular document stores include **MongoDB** and **Firestore**. They provide a flexible schema and support for complex queries.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

> The key difference between key-value stores and document stores is that while key-value stores map a single key to a single value, document stores allow for more complex data structures by storing entire documents as values.

</div>
<div class="c2" style = "width: 50%">

<img src="https://storage.googleapis.com/slide_assets/document-data-model.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Moniruzzaman</p>

</div>
</div>

<!--s-->

<div class="header-slide">

# Graph Databases

</div>

<!--s-->

## Graph Databases

Graph databases are designed to store and query data with complex relationships. They use graph structures, consisting of nodes (entities) and edges (relationships), to represent data. Graph databases are ideal for applications that require traversing relationships, such as social networks, recommendation systems, and fraud detection.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

One of the most popular graph databases is **Neo4j**.

</div>
<div class="c2" style = "width: 50%">

<img src="https://memgraph.com/images/blog/what-is-a-graph-database/Graph%20vs%20Relational%20DB.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Memgraph 2023</p>

</div>
</div>

<!--s-->

## L.02 | Q.02

Which database is more appropriate to store unstructured data, such as a blog post with text, images, and comments?

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; margin-left: 5%">

A. Relational Database

B. Document Store

C. Graph Database

</div>

<div class="c2" style = "width: 50%">

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L.02 | Q.02" width = "100%" height = "100%"></iframe>

</div>

<!--s-->

<div class="header-slide">

# Wide-Column Stores

</div>

<!--s-->

## Wide-Column Stores

Wide-column stores are a type of database that store data in columns rather than rows. This allows for efficient storage and retrieval of large volumes of data, making wide-column stores suitable for big data analytics and applications that require high write and read throughput.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Popular wide-column stores include **Apache Cassandra** and **Google Bigtable**.

</div>
<div class="c2" style = "width: 50%">

<img src="https://storage.googleapis.com/slide_assets/wide-column.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Le Blanc 2020</p>

</div>
</div>

<!--s-->

<div class="header-slide">

# In-Memory Databases

</div>

<!--s-->

## In-Memory Databases

In-memory databases store data in the main memory (RAM) rather than on disk, allowing for extremely fast access and processing. They are ideal for real-time analytics, caching, and applications that require low-latency data access. In-memory databases are often used in financial services and gaming.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Popular in-memory databases include **Redis** and **Memcached**.

</div>
<div class="c2" style = "width: 50%">

<img src="https://www.loginradius.com/blog/static/4eda1ce5a0f541d97fdf27cd88bf2a49/03979/index.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Redis 2025</p>

</div>
</div>

<!--s-->

<div class="header-slide">

# Time-Series Databases

</div>

<!--s-->

## Time-Series Databases

Time-series databases are designed to store and query time-stamped data, and provide specialized features for handling time-based queries and aggregations. This makes them ideal for applications that generate large volumes of time-based data, such as IoT devices, financial markets, and monitoring systems.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Time-series databases include **InfluxDB** and **TimescaleDB**.

</div>
<div class="c2" style = "width: 40%">

<img src="https://images.ctfassets.net/ilblxxee70tt/2vWtvIwvIowOgRLEN3VqtX/df58dfca4e190427fdb8e47034bcc3a6/indicators_desktop.jpg" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Robinhood Legend</p>

</div>
</div>

<!--s-->

## L.02 | Q.03

Let's say you have an application where latency is a critical factor, and you need to store and retrieve data as quickly as possible. Which type of database would be the best choice for this use case?

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; margin-left: 5%">

A. Relational Database

B. In-Memory Database

C. Time-Series Database

D. Wide-Column Store

</div>

<div class="c2" style = "width: 50%">

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L.02 | Q.03" width = "100%" height = "100%"></iframe>

</div>


<!--s-->

<div class="header-slide">

# Object-Oriented Databases

</div>

<!--s-->

## Object-Oriented Databases

Object-oriented databases store data as objects. An object is a data structure that contains both data and methods. It is a self-contained unit that can be manipulated and interacted with. This allows for complex data structures and relationships to be easily represented and manipulated. Object-oriented databases are often used in applications that require complex data modeling, such as computer-aided design (CAD) and multimedia applications.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Popular object-oriented databases include **db4o** and **ObjectDB**.

</div>
<div class="c2" style = "width: 40%">

<img src="https://storage.googleapis.com/slide_assets/OOBDB.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">IntelliPaat 2024</p>

</div>
</div>

<!--s-->

<div class="header-slide">

# Spatial Databases

</div>

<!--s-->

## Spatial Databases

Spatial databases are designed to store and query geospatial data, such as maps, geographic information systems (GIS), and location-based services. They provide specialized data types and indexing methods for efficiently handling spatial queries.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Popular spatial databases include **PostGIS** (an extension of PostgreSQL) and **Oracle Spatial**.

</div>
<div class="c2" style = "width: 50%">

<img src="https://www.oracle.com/a/ocom/img/rc24-what-is-geospatial-database-fig-1.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Oracle</p>

</div>
</div>

<!--s-->

<div class="header-slide">

# Blob Datastore

</div>

<!--s-->

## Blob Datastore

Blob datastores are designed to store unstructured data, such as images, videos, and documents. They provide a simple interface for uploading, retrieving, and managing large binary objects (blobs). Blob datastores are often used in applications that require large-scale storage of media files (like images and videos) as well as backups.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Popular blob datastores include **Amazon S3** and **Google Cloud Storage**.

</div>
<div class="c2" style = "width: 50%">

<img src="https://stensul.com/wp-content/uploads/2021/07/amazon-s3.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">AWS</p>

</div>
</div>

<!--s-->

## Blob Datastore Application: Image / Video Datasets

<div class = "col-wrapper"  style="font-size: 0.9em;">
<div class="c1" style = "width: 50%; margin-right: 3%">

### Chunking

Break large datasets into smaller chunks to allow for parallel processing and efficient loading. 

### Compression

Use compression techniques to reduce the size of the datasets stored in the blob datastore.

### Metadata Management
Store metadata about the datasets, such as file size, format, and creation date, to facilitate efficient querying and retrieval.

</div>
<div class="c2" style = "width: 50%">


### Versioning
Implement versioning for datasets to keep track of changes and updates. 

### Compute Resources
Distribute compute resources to process the datasets in parallel. This can be done using cloud services like AWS Lambda or Google Cloud Functions.

</div>
</div>

<!--s-->

## Blob Datastore Application: Efficiently Streaming Videos

<div class = "col-wrapper" style="font-size: 0.9em;">
<div class="c1" style = "width: 50%; margin-right: 5%">

### Chunking
Break the video into smaller chunks to allow for progressive loading and reduce latency. One common format for chunking is **HLS (HTTP Live Streaming)**.

### Adaptive Bitrate Streaming
Use adaptive bitrate streaming to adjust the video quality based on the user's network conditions. This ensures a smooth playback experience without buffering.

</div>
<div class="c2" style = "width: 50%">

### Content Delivery Network (CDN)
Use a CDN to cache and deliver the video content closer to the user. This reduces latency and improves the streaming experience.

### Caching
Implement caching strategies to store frequently accessed video chunks in memory or on disk.

</div>
</div>

<!--s-->

## L.02 | Q.04

Let's say you need to store a large amount of unstructured data, such as images and videos, and you don't really care about the structure of the data. Which type of database would be the best choice for this use case?

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; margin-left: 5%">

A. Relational Database

B. Blob Datastore

C. Key-Value Store

D. Document Store

</div>

<div class="c2" style = "width: 50%">

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L.02 | Q.04" width = "100%" height = "100%"></iframe>

</div>

<!--s-->

<div class="header-slide">

# Ledger Databases

</div>

<!--s-->

## Ledger Databases

Ledger databases are designed to store immutable, tamper-proof data, making them ideal for applications that require a secure and auditable record of transactions. They are often used in financial services (e.g., cryptocurrency) and supply chain management.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Popular ledger databases include **Amazon Quantum Ledger Database** and **Hyperledger Fabric**.

</div>
<div class="c2" style = "width: 50%">

<img src="https://storage.googleapis.com/slide_assets/AWS-Ledger.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">AWS</p>

</div>
</div>

<!--s-->

<div class="header-slide">

# Hierarchical Databases

</div>

<!--s-->

## Hierarchical Databases

Hierarchical databases store data in a tree-like structure, where each record has a parent-child relationship with other records. They are often used in applications that require a clear hierarchy, such as file systems and XML data. Hierarchical databases are less common today but were widely used in early database systems. They differ from graph databases in that they have a strict hierarchy, while graph databases allow for more complex relationships. 


<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Popular hierarchical databases include **IBM Information Management System (IMS)** and **Windows Registry**.

</div>
<div class="c2" style = "width: 50%">

<img src="https://cdn.prod.website-files.com/6064b31ff49a2d31e0493af1/674edbeac707e6fbef6eb561_AD_4nXdBPFPVx6dt5SpfGAXWJR5N_KCSwBgGzjxlSLfZwi7CEPq4NpnYxuh4FK8OvcdTvM0zcVOmcx7AFctw_m8NLpXJ0QDPL3MWMPVd54Hwl4Y1n5aP-tUqhU5_QxqNbXWIlnjOXBdOaQ.png" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Airbyte 2024</p>

</div>
</div>

<!--s-->

<div class="header-slide">

# Vector Databases

</div>

<!--s-->

## Vector Databases

Vector databases are designed to store and query high-dimensional data, such as embeddings from machine learning models. They are ideal for applications that require similarity search and recommendation systems, such as image and text search. We will go into more details later in the quarter on how these vector databases work.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Popular vector databases include **Pinecone** and **pgvector** (an extension of PostgreSQL).

</div>
<div class="c2" style = "width: 50%">

<img src="https://a.storyblok.com/f/219851/2188x1406/dff374c348/indexing-in-vector-database.webp" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Singh 2023</p>

</div>
</div>

<!--s-->

<div class="header-slide">

# Embedded Databases

</div>

<!--s-->

## Embedded Databases

Embedded databases are lightweight databases that are integrated into applications, allowing for local data storage and retrieval. They are often used in mobile apps, IoT devices, and applications that require offline access to data.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Popular embedded databases include **SQLite** and **Realm**.

</div>
<div class="c2" style = "width: 40%">

<img src="https://upload.wikimedia.org/wikipedia/commons/3/38/SQLite370.svg" width="100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">SQLite</p>


</div>
</div>

<!--s-->

<div class="header-slide">

# Summary

</div>

<!--s-->

## Summary

- A database is a collection of data that is organized in a way that allows for efficient retrieval and manipulation.

- A good database starts with a data model, enforces data integrity, provides efficient data access, implements robust security measures, and is designed to scale.

- There are many different types of databases, each suited for different use cases and data models.

<div style="overflow-y: auto; max-height: 200px; font-size: 0.7em; padding-left: 10%">

| Type | Description | Use Cases | Examples |
| --- | --- | --- | --- |
| Relational | Structured data with relationships | Transactional applications | MySQL, PostgreSQL |
| Key-Value | Simple key-value pairs | Caching, session management | Redis, DynamoDB |
| Document | Semi-structured data in documents | Content management, e-commerce | MongoDB, Couchbase |
| Graph | Data with complex relationships | Social networks, recommendation systems | Neo4j |
| Wide-Column | Large-scale data with flexible schema | Big data analytics | Cassandra, BigTable |
| In-Memory | Fast access to data in memory | Real-time analytics, caching | Redis, Memcached |
| Time-Series | Data with time-based relationships | IoT, financial data | InfluxDB, TimescaleDB |
| Object-Oriented | Data as objects with methods | CAD, multimedia applications | db4o, ObjectDB |
| Spatial | Geospatial data and queries | GIS, location-based services | PostGIS (PostgreSQL) |
| Blob Datastore | Unstructured data storage | Media files, backups | Amazon S3, Google Cloud Storage |
| Ledger | Immutable, tamper-proof data | Financial transactions, supply chain | Amazon Quantum Ledger Database |
| Hierarchical | Tree-like data structure | File systems, XML data | IBM IMS, Windows Registry |
| Vector | High-dimensional data for ML | Recommendation systems, image search | Pinecone, pgvector (PostgreSQL) |
| Embedded | Lightweight databases for applications | Mobile apps, IoT devices | SQLite, Realm |

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with navigating database options & basic SQL?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>
