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
  ## L.03 | Accessing Databases, Permissions, and Security

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 10%">

  <iframe src="https://lottie.host/embed/216f7dd1-8085-4fd6-8511-8538a27cfb4a/PghzHsvgN5.lottie" height = "100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Welcome to Data Intensive Systems.
  ## Please check in by creating an account and entering the code on the chalkboard.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

## Announcements

- Datacamp is active. Click [here](https://www.datacamp.com/groups/shared_links/322df3271c3d066625f90f3390391d038ac896168157a0484980012c7e835a24) to access the classroom.
- We'll start today by picking up where we left off on [L.02](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L02/#/46)
- H.02 will be released today and is due in a week (4/17).

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with the following: 

  - Accessing Databases
  - Managing Permissions on Databases
  - Security Best Practices on Databases

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

## Agenda

1. **Accessing Databases**
    - Connecting to databases
    - Querying databases
    - SQL Example: Connecting to a PostgreSQL database through CLI (psql)
    - REST API Example: Connecting to a pinecone database through Python (pinecone)

2. **Permissions**
    - User roles and permissions
    - Granting and revoking access
    - Example: Setting up user roles in PostgreSQL

3. **Security**
    - Essential security practices

<!--s-->

<div class="header-slide">

# Accessing Databases

</div>

<!--s-->

## Connecting to Databases

Recall that a database is a collection of data that is organized in a way that allows for efficient retrieval and manipulation. At it's core, a database is simply a structured way to **store** and **manage** data on a computer. A database can be as simple as a text file or a spreadsheet, or as complex as a distributed system that spans multiple servers and locations. But at it's core, all databases run on computers / servers that you can connect to.

<div style="text-align: center;">
    <img src="https://guide-images.cdn.ifixit.com/igi/BCU4AgbFicGvFcZA.large" width="40%" style="border-radius: 10px;" />
    <p style="text-align: center; font-size: 0.6em; color: grey;">iFixit</p>
</div>

<!--s-->

## Connecting to Databases

To access a database from a client application, you need to establish a connection to the database server. Applications exist in a variety of programming languages and frameworks, and each has its own way of connecting to databases. Here are some common ways to connect to databases:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Command Line Interface (CLI)
A text-based interface for interacting with databases. Examples include <span class="code-span">psql</span> for PostgreSQL and <span class="code-span">mysql</span> for MySQL.

### Database Drivers (Python)
Libraries that allow you to connect to databases from your application code. Examples include <span class="code-span">psycopg2</span> for PostgreSQL and <span class="code-span">mysql-connector-python</span> for MySQL.

</div>
<div class="c2" style = "width: 50%">

### REST APIs
Web-based interfaces that allow you to interact with databases over HTTP. Examples include the Pinecone API for vector databases and the Firebase API for NoSQL databases.

</div>
</div>

<!--s-->

## L.03 | Q.01

Client side applications are designed to run on the client's computer, while server side applications are designed to run on the server. Let's say you have an application running in a client's **browser**. How would you connect to a database from this application?

<div class = 'col-wrapper'>
<div class='c1' style = 'width: 50%; margin-left: 5%; margin-top: 5%'>

A. Use a database driver like <span class="code-span">psycopg2</span> to connect to the database.</br></br>
B. Use a REST API to connect to the database.</br></br>
C. Use a CLI tool like <span class="code-span">psql</span> to connect to the database.

</div>
<div class='c2' style = 'width: 50%;'>
  <iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.03 | Q.01' width = '100%' height = '100%'></iframe>
</div>
</div>


<!--s-->

## Connecting to Databases | CLI

One common way to connect to databases is through the Command Line Interface (CLI). The CLI allows you to interact with the database using text-based commands.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Logging In

Here's an example of connecting to a PostgreSQL database hosted on a remote server using the <span class="code-span">psql</span> CLI. 

```bash
$ psql -h mydatabase.com -u myuser -d mydatabase
```

Where <span class="code-span">-h</span> specifies the hostname of the database server, <span class="code-span">-u</span> specifies the username, and <span class="code-span">-d</span> specifies the name of the database.

</div>
<div class="c2" style = "width: 50%">

### Executing SQL Queries

After logging in, you can execute SQL queries and commands directly from the CLI.

```sql
mydatabase=> SELECT * FROM users;
```

</div>
</div>

<!--s-->

## Connecting to Databases | Database Drivers (Python)

Another common way to connect to databases is through database drivers. Database drivers are libraries that allow you to connect to databases from your application code. Below is an example of connecting to a PostgreSQL database using the <span class="code-span">psycopg2</span> library in Python.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### PostgreSQL

```python
import psycopg2

# Connect to an existing database
conn = psycopg2.connect(
    dbname="mydatabase",
    host="mydatabase.com",
    user="myuser",
    password="mypassword"
)

# Open a cursor to perform database operations
cur = conn.cursor()

# Execute a SQL query
cur.execute("SELECT * FROM users;")
rows = cur.fetchall()

# Close communication with the database
cur.close()
conn.close()

```

</div>
<div class="c2" style = "width: 50%">

### PostgreSQL & Pandas

```python
import pandas as pd

# Connect to an existing database
conn = psycopg2.connect(
    dbname="mydatabase",
    host="mydatabase.com",
    user="myuser",
    password="mypassword"
)

# Read data from the database
df = pd.read_sql_query("SELECT * FROM users;", conn)

# Close the connection
conn.close()

```

</div>
</div>

<!--s-->

## Connecting to Databases | REST APIs

REST APIs are web-based interfaces that allow you to interact with databases over HTTP. Below is an example of connecting to a Pinecone vector database using the Pinecone API in Python, and a Firebase NoSQL database using the Firebase API in Python.

<div class = "col-wrapper">

<div class="c1" style = "width: 50%">

### Pinecone (Vector DB)

```python
import pinecone

# Connect to the Pinecone service
pinecone.init(
    api_key="my_api_key",
    environment="us-west1-gcp"
)

# Query the database
results = pinecone.query(
    index_name="my_index",
    query_vector=[0.1, 0.2, 0.3]
)

```

</div>

<div class="c2" style = "width: 50%">

### Firebase (NoSQL DB)

```python
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize the Firebase Admin SDK
cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# Get a reference to the Firestore database
db = firestore.client()

# Query the database
docs = db.collection("users").get()

```

</div>

</div>

<!--s-->

<div class="header-slide">

# Permissions

</div>

<!--s-->

## User Roles and Permissions

In a database management system, user roles and permissions are used to control access to the database and its objects. User roles define the privileges that a user has, while permissions specify the actions that a user can perform on the database objects.

This is a critical aspect of database security, as it ensures that only authorized users can access and modify the data in the database.

<!--s-->

## User Roles and Permissions

Here are some common user roles and permissions that are typically defined in a database management system:

| Role | Description |
|---------------|------------- |
| Admin     | Full access to all database objects and operations. |
| User      | Limited access to specific database objects and operations.       |
| Read-only | Read-only access to the database, with no ability to modify data. |
| Write-only| Write-only access to the database, with no ability to read data.  |
| Backup    | Access to backup and restore operations.        |
| DBA       | Database Administrator role with full control over the database.  |
| Developer | Access to development tools and operations.     |
| Auditor   | Access to audit logs and monitoring tools.      |
| Security  | Access to security settings and configurations. |
| Guest     | Limited access for temporary or guest users.    |
| Public    | Public access for anonymous users.     |
| Custom    | Custom roles and permissions defined by the database administrator. |

<!--s-->

## Granting and Revoking Access

In a database management system, access to database objects is granted or revoked using the <span class="code-span">GRANT</span> and <span class="code-span">REVOKE</span> commands. These commands are used to assign or remove specific privileges to users or roles.

Here is an example of **granting** read-only access to the "houses" table in a database for a user (myuser):

```sql
-- Grant SELECT privilege on the houses table to the user myuser.
GRANT SELECT ON houses TO myuser;
```

Here is an example of **revoking** read-only access to the "houses" table in a database for a user (myuser):

```sql
-- Revoke SELECT privilege on the houses table from the user myuser.
REVOKE SELECT ON houses FROM myuser;
```

<!--s-->

## Granting and Revoking Access

Here is a cheatsheet for granting and revoking access in a database:

| Command | Description |
|---------------|------------- |
| <span class="code-span">GRANT SELECT</span> | Grants read-only access to a table. |
| <span class="code-span">GRANT INSERT</span> | Grants permission to insert rows into a table. |
| <span class="code-span">GRANT UPDATE</span> | Grants permission to update rows in a table. |
| <span class="code-span">GRANT DELETE</span> | Grants permission to delete rows from a table. |
| <span class="code-span">GRANT ALL</span> | Grants all privileges on a table. |
| <span class="code-span">REVOKE SELECT</span> | Revokes read-only access to a table. |
| <span class="code-span">REVOKE INSERT</span> | Revokes permission to insert rows into a table. |
| <span class="code-span">REVOKE UPDATE</span> | Revokes permission to update rows in a table. |

<!--s-->

## Granting and Revoking Access | User Roles

Often, we don't want to grant permissions to individual users, but rather to groups of users. This is where roles come in. Roles are a way to group users together and assign permissions to the group as a whole.

Here is an example of creating a custom role (myrole) in a database and granting read-only access to the houses table to that role:

```sql
-- Create a custom role myrole.
CREATE ROLE myrole;

-- Grant SELECT privilege on the houses table to the role myrole.
GRANT SELECT ON houses TO myrole;
```

Then, you can assign users to the role using the <span class="code-span">GRANT</span> command:

```sql
-- Assign users to the role myrole.
GRANT myrole TO myuser;
GRANT myrole TO myuser2;
GRANT myrole TO myuser3;

```

<!--s-->

## L.03 | Q.02

What is the difference between a user role and a user permission?

<div class = "col-wrapper">
<div class='c1' style = 'width: 50%; margin-left: 5%; margin-top: 10%'>

A. A user role is a group of users, while a user permission is a specific action that a user can perform.<br><br>
B. A user role is a specific action that a user can perform, while a user permission is a group of users.<br>

</div>
<div class="c2" style = "width: 50%">

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L.03 | Q.02" width = "100%" height = "100%"></iframe>

</div>
</div>

<!--s-->

<div class="header-slide">

# Security

</div>

<!--s-->

## Essential Security Practices

Security is a critical aspect of database management. Here are some essential security practices that you should follow to protect your databases and data. Many platforms will have built-in security features, but it's important to understand the basics so that you can configure them correctly.

1. **Encryption**: Encrypt data at rest and in transit to protect it from unauthorized access.

2. **Backups**: Regularly backup your databases to prevent data loss in case of hardware failure or other disasters.

3. **Access Control**: Implement access controls to restrict access to sensitive data and operations.

4. **Monitoring**: Monitor your databases for suspicious activities and unauthorized access.

5. **Auditing**: Keep audit logs of database activities to track changes and identify security incidents.

<!--s-->

## Essential Security Practices | Encryption

Encryption is the process of encoding data so that only authorized users can access it.

1. **Data Encryption**: Encrypting data at rest to protect it from unauthorized access. This can be done using encryption algorithms and keys.

2. **Transport Encryption**: Encrypting data in transit to protect it as it travels between the client and the server. This can be done using SSL/TLS protocols.

<!--s-->

## Essential Security Practices | Backups

Backups are copies of your database that you can use to restore your data in case of hardware failure, data corruption, or other disasters.

1. **Regular Backups**: Schedule regular backups of your databases to ensure that you have up-to-date copies of your data.

2. **Offsite Backups**: Store backups in a secure offsite location to protect them from local disasters.

3. **Automated Backups**: Use automated backup tools to simplify the backup process and ensure that backups are performed consistently.

4. **Testing Backups**: Regularly test your backups to ensure that they are valid and can be restored successfully.

<!--s-->

## Essential Security Practices | Access Control

Access control is the process of restricting access to sensitive data and operations in your database.

1. **Role-Based Access Control**: Use roles to group users together and assign permissions to the group as a whole.

2. **Least Privilege Principle**: Grant users the minimum level of access required to perform their tasks.

3. **Strong Passwords**: Enforce strong password policies to prevent unauthorized access.

4. **Multi-Factor Authentication**: Use multi-factor authentication to add an extra layer of security to user accounts.

<!--s-->

## Essential Security Practices | Monitoring

Monitoring is the process of tracking and analyzing database activities to detect suspicious behavior and unauthorized access.

1. **Real-Time Monitoring**: Monitor your databases in real-time to detect and respond to security incidents quickly.

2. **Alerting**: Set up alerts for unusual activities, such as failed login attempts or unauthorized access.

3. **Log Management**: Keep audit logs of database activities to track changes and identify security incidents.

4. **Incident Response**: Have an incident response plan in place to respond to security incidents effectively.

<!--s-->

## Essential Security Practices | Auditing

Auditing is the process of tracking and recording database activities to ensure compliance with security policies and regulations.

1. **Audit Logs**: Keep detailed audit logs of database activities, including user logins, queries, and modifications.

2. **Log Retention**: Retain audit logs for a specified period to comply with security policies and regulations. This may include GDPR, HIPAA, or other regulations.

3. **Log Analysis**: Regularly analyze audit logs to identify security incidents and compliance issues.

4. **Reporting**: Generate reports on database activities and security incidents to track trends and improve security practices.

<!--s-->

<div class="header-slide">

# Summary

</div>

<!--s-->

## Summary

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Accessing Databases
Establish connections to your databases using CLI tools, database drivers, and REST APIs.

### Permissions
Create and manage user roles to control access to the database. This will involve granting and revoking permissions to ensure that only authorized users can access and modify the data.

</div>
<div class="c2" style = "width: 50%">

### Security
Follow essential security practices to protect your databases from unauthorized access and data breaches. This includes encrypting data at rest and in transit, performing regular backups, monitoring database activities, and maintaining audit logs.

</div>
</div>


<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with the following: 

  - Accessing Databases
  - Managing Permissions on Databases
  - Security Best Practices on Databases

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # H.02 | SQL Practice
  ## Due: 4/17

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%;">

  <img src="https://docs.snowflake.com/en/_images/sample-data-tpch-schema.png" alt="Snowflake Schema" style = "border-radius: 10px">
  </div>
</div>

<!--s-->

## H.02 Instructions

We'll do the first SQL query together to make sure everyone is set up.

1. Navigate to the homework directory in the class repo.

2. If you don't already have <span class="code-span">.env</span> filled out with your credentials, please do so now using the [L.01](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L01/#/16) instructions.

3. Run <span class="code-span">git pull</span> in the terminal to get the latest version of the homework.
4. Open <span class="code-span">homeworks/sql_practice.ipynb</span> to get started.


