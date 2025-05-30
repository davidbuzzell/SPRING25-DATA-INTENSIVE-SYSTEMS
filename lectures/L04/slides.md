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
  ## L.04 | OLAP Systems

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

- We're going to pivot away from the Docker homework setup.
  - The first 10 minutes of this lecture will be dedicated to adding this setup and removing Docker from your system.

- ~ 50% of you have already submitted [H.02](https://github.com/drc-cs/SPRING25-DATA-INTENSIVE-SYSTEMS/blob/main/homeworks/H02/sql_practice.ipynb). 
  - Please submit by **Thursday @ 11:59 PM**.

- [Syllabus](https://canvas.northwestern.edu/courses/231041) has been updated with times and locations for make-up lectures.
  - This Wednesday @ 10:30A in Kellogg Global Hub L110.

- Office hours updates
  - Thursdays (D'Arcy) from 2:15 PM to 3:15 PM in Mudd 3510, or by appointment.
  - Fridays (Mo) from 3:00 PM to 4:00 PM in Mudd First Floor Lobby. 

<!--s-->

## Announcements | Lecture Efficacy Overview

Thank you everyone for doing the lecture polls!

| Lecture |  Avg. Starting Score (/5) |Avg Delta [25th-75th Percentile] | P-Value  |
|------------|-----------------|-------|-------|
| L.01 | 2.54 | <span style="color:#2eba87; font-weight: bold">+ 0.75 [0.0-1.0]</span>  | $7.7 \times 10^{-3}$ | 2.54 |
| L.02 | 2.66 | <span style="color:#2eba87; font-weight: bold">+ 0.83 [0.0-1.0]</span>  | $1.0 \times 10^{-4}$ | 2.66 |
| L.03 | 2.24 | <span style="color:#2eba87; font-weight: bold">+ 1.18 [1.0-2.0]</span>  | $1.54 \times 10^{-6}$ |

<!--s-->

## Homework Setup

<div style="overflow-y: scroll; font-size: 0.8em;">

1. If you don't already have a Github account, [create one](https://github.com/) and log in. Then go to this [public repo](https://github.com/drc-cs/SPRING25-DATA-INTENSIVE-SYSTEMS) for our course.

2. Click on the green button "Code". 

3. Go to the Codespaces tab.

4. Click "Create codespace on main". This will navigate you to a codespace instance with vscode installed.

5. Give the codespace at least **3 minutes** to load. It should automatically install the requirements for this course. When you see a <span class="code-span">venv/</span> folder appear, you can continue.

6. Navigate to the <span class="code-span">homeworks/.env/</span> file and add your credentials like we did in class.

7. Navigate to <span class="code-span">homeworks/H01/hello_world.ipynb</span>.

8. Select Kernel in the top right. Install the suggested extensions.

9. Select Kernel again, and this time you should also select "Python Environments"

10. Select <span class="code-span">venv</span>.

11. Update the string in the first cell to equal 'Hello, World!'

12. Run the cells. This will re-submit your H.01, verifying everything is working as expected. 

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with **OLAP** concepts?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# L.04 | Online Analytical Processing (OLAP)

</div>

<!--s-->

## Agenda

1. ### What is Online Analytical Processing (OLAP)?
    - Online Transactional Processing (OLTP) Recap
    - OLTP vs OLAP
    - OLAP Solutions
2. ### Rise and Fall of the OLAP Cube
    - OLAP Cube Definition
    - OLAP Cube Operations
3. ### Columnar Databases
    - Why Column-Based Databases?
    - Advances that make Columnar DBs feasible

<!--s-->

## What is OLTP?

**Online Transaction Processing (OLTP)** is a class of software applications capable of supporting transaction-oriented programs. OLTP systems are designed to manage a large number of short online transactions (INSERT, UPDATE, DELETE).

JOIN operations are common in OLTP systems, however, they are **expensive** operations.

<div class = "col-wrapper">
  <div class="c1">

  ### Characteristics of OLTP:

  - High transaction volume
  - Short response time
  - Data integrity
  - Normalized data

  </div>
  <div class="c2" style="width: 50%; height: auto;">
  <img style="border-radius: 10px;" src="https://planetscale.com/assets/blog/content/schema-design-101-relational-databases/db72cc3ac506bec544588454972113c4dc3abe50-1953x1576.png" />
<p style="text-align: center; font-size: 0.6em; color: grey;">Ramos 2022</p>

  </div>
</div>

<!--s-->

## L.04 | Q.01

What is **normalized** data in the context of OLTP systems?

<div class = 'col-wrapper'>
<div class='c1' style = 'width: 50%; margin-left: 5%; margin-top: 10%;'>

A. Data that is stored in a way that **minimizes** redundancy.<br><br>
B. Data that is stored in a way that **maximizes** redundancy.

</div>
<div class='c2' style = 'width: 50%;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.04 | Q.01' width = '100%' height = '100%'></iframe>
</div>
</div>

<!--s-->

## OLTP vs OLAP

<div class = "col-wrapper">
<div class="c1" style="width: 50%; height: auto;">

### Online **Transaction** Processing (OLTP)
OLTP is designed for managing transaction-oriented applications.

</div>

<div class="c2" style="width: 50%; height: auto;">

### Online **Analytical** Processing (OLAP)

OLAP is designed for data analysis and decision-making.

</div>
</div>

<!--s-->

## OLTP vs OLAP

| Feature | OLTP | OLAP |
|---------|------|------|
| Purpose | Transaction processing | Data analysis |
| Data Model | Usually Normalized | Usually Denormalized |
| Queries | Simple, short | Complex, long |
| Response Time | Short | Long |
| Data Updates | Frequent | Infrequent |

<!--s-->

## What is OLAP?

OLAP is an approach to answer multi-dimensional analytical queries swiftly. OLAP allows analysts, managers, and executives to gain insight through rapid, consistent, and interactive access to a wide variety of possible views of data.

<div class = "col-wrapper">
<div class="c1" style="width: 50%; height: auto;">

### Characteristics of OLAP:

- Designed for complex queries
- Read-heavy workloads

</div>

<div class="c2" style="width: 50%; height: auto;">

<img style="border-radius: 10px;" src="https://i0.wp.com/olap.com/wp-content/uploads/2019/06/olap-3d-cube.png?fit=2016%2C1890&ssl=1" />
<p style="text-align: center; font-size: 0.6em; color: grey;">olap.com 2025</p>

</div>
</div>

<!--s-->

## Database Schemas

A database schema is the skeleton structure that represents the logical view of the entire database. It defines how data is organized and how relationships between data are handled. 

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

A schema optimized for OLTP (e.g. normalized) will not perform well for OLAP workloads which are read-heavy. This is because OLTP scemas can lead to complex queries that require multiple joins, which can be slow for analytical workloads.

</div>
<div class="c2" style = "width: 50%">

<img src = "https://cdn-ajfbi.nitrocdn.com/GuYcnotRkcKfJXshTEEKnCZTOtUwxDnm/assets/images/optimized/rev-c2378d8/www.astera.com/wp-content/uploads/2024/05/Database-schema.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Asteria 2024</p>

</div>
</div>

<!--s-->

## Database Schemas

The **Star Schema** is a type of database schema that is optimized for data warehousing and OLAP applications. It consists of a central fact table surrounded by dimension tables. Star Schemas are typically denormalized, meaning they contain redundant data to optimize read performance.

<div class="col-wrapper col-centered">
  <img src="https://cdn.prod.website-files.com/5e6f9b297ef3941db2593ba1/614df58a1f10f92b88f95709_Screenshot%202021-09-24%20at%2017.46.51.png" style="border-radius: 10px; height: 60%;" />
  <p style="text-align: center; font-size: 0.6em; color: grey;">Asteria 2024</p>
</div>

<!--s-->

## L.04 | Q.02

Consider the following database schemas. Which is better suited for OLAP?

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

A.

<img style="border-radius: 10px; height: 30%;" src="https://planetscale.com/assets/blog/content/schema-design-101-relational-databases/db72cc3ac506bec544588454972113c4dc3abe50-1953x1576.png" />

B.

<img src="https://cdn.prod.website-files.com/5e6f9b297ef3941db2593ba1/614df58a1f10f92b88f95709_Screenshot%202021-09-24%20at%2017.46.51.png" style="border-radius: 10px; height: 30%;" />

</div>
<div class="c2" style = "width: 50%">

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L.04 | Q.02" width = "100%" height = "100%"></iframe>

</div>
</div>

<!--s-->

<div class = "header-slide">

# Rise and Fall of the OLAP Cube

</div>

<!--s-->

## OLAP Cube

An **OLAP Cube** is a multi-dimensional array of data used in business intelligence. Instead of storing data in a tabular format, cubes allow for complex calculations, trend analysis, and sophisticated data modeling. This is because OLAP cubes can store data in multiple dimensions, allowing users to analyze data from different perspectives without the need for complex joins.

<div class = "col-wrapper" style = "margin: 0px; padding: 0px">
<div class="c1" style = "width: 50%; margin: 0px; padding: 0px">

A common workflow to build OLAP cubes:

1. **Extract**: Data is extracted from various sources, such as databases, spreadsheets, or flat files.
2. **Transform**: Data is cleaned, transformed, and (often) aggregated.
3. **Load**: Data is loaded into the OLAP cube.

<div style="font-size: 0.6em; margin: 0; padding: 0;">

> Note: This is often referred to as the ETL process, and is not specific to OLAP / OLAP cubes.
</div>

</div>
<div class="c2" style = "width: 50%">

<div class = "col-wrapper col-centered">
<img style="border-radius: 10px;" src="https://i0.wp.com/olap.com/wp-content/uploads/2019/06/olap-3d-cube.png?fit=2016%2C1890&ssl=1" />
<p style="text-align: center; font-size: 0.6em; color: grey;">olap.com 2025</p>
</div>

</div>
</div>


<!--s-->

## OLAP Cube Definition

<div class = "col-wrapper">
<div class="c1" style="width: 50%; height: auto;">

We can define OLAP as a function of axes. Consider a data cube with dimensions $(X, Y, Z)$. An OLAP operation can be represented as:

$$ f : (X, Y, Z) \rightarrow \text{W} $$

Where the result (W) is a subset or aggregation of the data based on the specified dimensions and measures.

</div>

<div class="c2" style="width: 50%; height: auto;">

<img style="border-radius: 10px;" src="https://i0.wp.com/olap.com/wp-content/uploads/2019/06/olap-3d-cube.png?fit=2016%2C1890&ssl=1" />
<p style="text-align: center; font-size: 0.6em; color: grey;">olap.com 2025</p>

</div>
</div>

<!--s-->

## L.04 | Q.03

What data is highlighted in orange?

<div class = "col-wrapper">
<div class="c1" style="width: 50%; height: auto; margin-left: 5%;">

A. Laptop sales in USA during Q1.<br><br>
B. TV sales in Asia during Q1.<br><br>
C. Laptop sales in Asia during Q1.

<img style="border-radius: 10px;" src="https://i0.wp.com/olap.com/wp-content/uploads/2019/06/olap-3d-cube.png?fit=2016%2C1890&ssl=1" width = "70%"/>

</div>

<div class="c2" style="width: 50%; height: auto;">

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L.04 | Q.03" width = "100%" height = "100%"></iframe>


</div>
</div>

<!--s-->

## OLAP Cube Operations

OLAP Cubes enable various operations to analyze data.

| Operation | Description | Example |
|-----------|-------------|---------|
| Drill-up | Aggregates data along a dimension | Monthly sales to quarterly sales |
| Drill-down | Decomposes data into finer levels | Quarterly sales to monthly sales |
| Slice | Selects a single dimension | Sales for a specific product |
| Dice | Selects two or more dimensions | Sales for a specific product in a specific region |

<!--s-->

## OLAP Cube Operations | Drill-down

Drill-down decomposes data into finer levels. 

Drilling down in *Outdoor protective equipment* reveals specific data inside the category (*insect repellent, Sunblock, First Aid*)

<img src = "https://upload.wikimedia.org/wikipedia/commons/9/9b/OLAP_drill_up%26down_en.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Wikipedia 2023</p>

<!--s-->

## OLAP Cube Operations | Drill-up

Drill-up aggregates data along a dimension. Drilling up in *Outdoor protective equipment* reveals the total sales for the entire category.

<img src = "https://upload.wikimedia.org/wikipedia/commons/9/9b/OLAP_drill_up%26down_en.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Wikipedia 2023</p>

<!--s-->

## OLAP Cube Operations | Slice

Slice selects a single dimension. Here we just want to see *2004* data.

<img src = "https://upload.wikimedia.org/wikipedia/commons/a/a6/OLAP_slicing_en.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Wikipedia 2023</p>

<!--s-->

## OLAP Cube Operations | Dice

Dice selects two or more dimensions. Here you can see diced to only read *Accessories* $\rightarrow$ *Golf equipment*.

<img src = "https://upload.wikimedia.org/wikipedia/commons/c/c7/OLAP_dicing_en.png" style="border-radius: 10px"/>

<p style="text-align: center; font-size: 0.6em; color: grey;">Wikipedia 2023</p>

<!--s-->

## OLAP Cubes | Downfall

In recent years, the use of OLAP cubes has declined due to:

- **Complexity**: Building and maintaining OLAP cubes can be complex and time-consuming.
- **Data Volume**: The explosion of data volume and variety makes it challenging to pre-aggregate data.
- **Real-time Analytics**: The need for real-time data access and analytics has led to the adoption of more flexible data architectures.

Still, many organizations continue to use OLAP cubes for specific use cases, especially in traditional business analytics environments.

<!--s-->

<div class = "header-slide">

# Columnar Databases

</div>

<!--s-->

## Columnar Databases

**Columnar Databases** store data tables primarily by column rather than row. This storage approach is ideal for OLAP scenarios as it dramatically speeds up the querying of large datasets.

<div class="col-wrapper col-centered">
<img src = "https://storage.googleapis.com/gweb-cloudblog-publish/images/BigQuery_Explained_storage_options_2.max-700x700.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Thallum, 2020</p>
</div>

<!--s-->

## Why Column-Based Databases?

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Row-based databases

Row-based databases store data in rows, which is efficient for transactional workloads but can be inefficient for analytical queries that often require scanning large amounts of data across multiple rows.

### Column-based databases

Column-based databases provide faster data retrieval and more effective data compression than traditional row-oriented databases, especially suited for read-oriented tasks.

</div>
<div class="c2 col-centered" style = "width: 50%">

<img src = "https://storage.googleapis.com/gweb-cloudblog-publish/images/BigQuery_Explained_storage_options_2.max-700x700.png" style="border-radius: 10px"/>
<p style="text-align: center; font-size: 0.6em; color: grey;">Thallum, 2020</p>

</div>
</div>

<!--s-->

## Why Column-Based Databases?

### Advantages of Columnar Storage

1. **Faster Query Performance**: Only the necessary columns are read, reducing I/O operations.

2. **Better Compression**: Similar data types are stored together, allowing for more efficient compression algorithms.

3. **Improved Analytics**: Columnar storage is optimized for analytical queries, making it easier to perform aggregations and calculations.

4. **Scalability**: Columnar databases can handle large volumes of data and scale horizontally by adding more nodes to the cluster.

<!--s-->

## Cloud-based Columnar Data Warehouse Services

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 5%;">

### AWS Redshift

Uses columnar storage, massively parallel processing, and optimized compression to enhance performance.

### GCP BigQuery

Serverless, highly scalable, and cost-effective multi-cloud data warehouse designed for business agility.

</div>
<div class="c2" style = "width: 50%">

### SnowFlake

Provides a unique architecture with a separation of compute and storage layers, allowing for scalable and elastic performance.

</div>
</div>



<!--s-->

## Example: Columnar DB Data Preprocessing
### Standardizing a Column using BigQuery

```python
from google.cloud import bigquery

# Initialize a BigQuery client
client = bigquery.Client()

# Define a query to standardize paw_size in a `puppies` table
query = "SELECT ML.STANDARD_SCALER(paw_size) AS paw_size_scaled FROM `project.dataset.puppies`"

# Load the query results into a Pandas DataFrame
df = client.query(query).to_dataframe()
```

<!--s-->

## Example: Columnar DB Data Preprocessing
### Standardizing a Column with Snowflake

```python
from connection import connect_to_snowflake
import pandas as pd

# Connect to Snowflake.
conn = connect_to_snowflake(database="doggos_db", schema="puppies_schema")
cursor = conn.cursor()

# Define a query to standardize paw_size in a `puppies` table.
query = """
SELECT
    paw_size,
    (paw_size - AVG(paw_size) OVER ()) / STDDEV_SAMP(paw_size) OVER () AS standardized_paw_size
FROM
    puppies;
"""

# Use pandas to read the query results into a DataFrame.
df = pd.read_sql(query, conn)

```

<!--s-->

<div class="header-slide">

# DIS Decision Making

</div>

<!--s-->

## L.04 | Q.04

<div style = "font-size: 0.9em;">

Let's imagine a scenario where you are a technical product manager at an early (Series B) startup in the textiles industry.

You're given a small dataset (~ 5MB) to analyze. It contains data on the ecological impact of your business from a single factory. You do not expect this dataset to grow over time or get more complicated -- in other words, this is a one-off project. You're asked to run a simple regression analysis on the data. What's is the most reasonable approach?

</div>

<div class = 'col-wrapper' style = "height: 100%";>
<div class='c1' style = 'width: 60%; margin-left: 5%; font-size: 0.9em;'>


A. Load the data into an OLAP system and do the regression on scalable infrastructure.<br><br>
B. Load the data into an OLTP system for storage and do the regression on your laptop.<br><br>
C. Load the data onto your laptop and just do the regression on your laptop. Save the data in a bucket somewhere.

</div>
<div class='c2' style='width: 50%; margin-top: 0px; padding-top: 0px;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.04 | Q.04' width = '100%' height = '60%'></iframe>
</div>
</div>

<!--s-->

## L.04 | Q.05

<div style = "font-size: 0.9em;">

Let's imagine a scenario where you are a technical product manager at an early (Series B) startup in the textiles industry.

You're given a small dataset (~ 5MB) to analyze. It contains data on the ecological impact of your business from a single factory. You **do** expect this dataset to grow over time and get more complicated -- in other words, this is **not** a one-off analysis project. You're asked to run a simple regression analysis on the data. What's is the most reasonable approach?

</div>

<div class = 'col-wrapper' style = "height: 100%";>
<div class='c1' style = 'width: 60%; margin-left: 5%; font-size: 0.9em;'>


A. Load the data into an OLAP system and do the regression on scalable infrastructure.<br><br>
B. Load the data into an OLTP system for storage and do the regression on your laptop.<br><br>
C. Load the data onto your laptop and just do the regression on your laptop. Save the data in a bucket somewhere.

</div>
<div class='c2' style='width: 50%; margin-top: 0px; padding-top: 0px;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.04 | Q.05' width = '100%' height = '60%'></iframe>
</div>
</div>

<!--s-->

## L.04 | Q.06

<div style = "font-size: 0.9em;">

Let's imagine a scenario where you are a technical product manager at rapidly scaled startup in the textiles industry. You're given an extremely large dataset (~ 5TB) to analyze. It contains data on the ecological impact of your business across the globe.

You're asked to run a simple regression analysis on the data. What's is the most reasonable approach?

</div>

<div class = 'col-wrapper' style = "height: 100%";>
<div class='c1' style = 'width: 60%; margin-left: 5%; font-size: 0.9em;'>


A. Load the data into an OLAP system and do the regression on scalable infrastructure.<br><br>
B. Load the data into an OLTP system for storage and do the regression on your laptop.<br><br>
C. Load the data onto your laptop and just do the regression on your laptop. Save the data in a bucket somewhere.

</div>
<div class='c2' style='width: 50%; margin-top: 0px; padding-top: 0px;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.04 | Q.06' width = '100%' height = '60%'></iframe>
</div>
</div>

<!--s-->

<div class="header-slide">

# Summary

</div>

<!--s-->

## Summary

### Online Analytical Processing (OLAP)
OLAP is a powerful approach to data analysis that enables users to interactively explore and analyze large datasets. 

### OLAP cubes
While once the gold standard for OLAP, have seen a decline in popularity due to their complexity and the rise of more flexible data architectures. 

### Columnar databases
Columnar databases have emerged as a key technology for OLAP, providing faster query performance and better compression.

<!--s-->

<div class="header-slide">

# OLAP Demo

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with **OLAP** concepts?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# H.02 | SQL Practice

</div>

<!--s-->