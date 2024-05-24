-- I wrote this SQL query as part of a larger set of SQL queries to analyze the Ashley Madison breach dataset.
-- I restored the attacker's SQL backups into a Docker container and began running
-- queries to explore the data. One of the things I was curious about was which
-- email domains had the highest number of Ashley Madison users.
-- My larger Ashley Madison GitHub repo has the other Python/Bash/SQL scripts, but it is private, so send me your GitHub username if you'd like access.
-- 
-- It uses regular expressions to extract the domain, which is trickier than
-- you think since some countries use multiple periods in the Top Level Domain.
--
-- I also created some new Indexes to speed up the searches and cut runtime from
-- 4 minutes down to 1 second.
--
-- Answer: besides the public email providers like Gmail, Hotmail, etc, the
-- most popular email domains are Facebook (employees), 
-- law enforcement officers, defense contrators, and especially the federal bureau of prisons.

-- create the new column for email domains, set default value to NULL
ALTER TABLE aminno_member_email
  ADD COLUMN email_domain VARCHAR(255) NULL;

-- see how many rows will need to get updated
-- query took 17 seconds
-- table has 36,397,896 rows
SELECT Count(*)
FROM   aminno_member_email;

-- see how many rows have mail addresses
-- query took 27 seconds
-- table has 36,396,146 rows that have an @ symbol in them
-- only 1,750 rows do not have an @ symbol in the email
-- 99.5% of the rows have an '@' symbol in the email
SELECT Count(*)
FROM   aminno_member_email
WHERE  email LIKE '%@%';

-- assign values for email domain if an '@' is in the email field
-- Query OK, 36396146 rows affected (51 min 8.85 sec)
-- Rows matched: 36396146  Changed: 36396146  Warnings: 0
UPDATE aminno_member_email
SET    email_domain = Trim(Lower(RIGHT(email, Length(email) - Locate('@', email)
                                 )))
WHERE  email LIKE '%@%';

-- create an index on email domain for faster queries in the future
CREATE INDEX index_email_domain ON aminno_member_email(email_domain);

-- took 5 min:
CREATE INDEX index_email ON aminno_member_email(email);

-- run a quick test to see if there are any email domains that aren't blank
-- took 4 minutes before the index 
-- 1 second after creating the index
SELECT email_domain
FROM   aminno_member_email
WHERE  email_domain <> ''
LIMIT  20;

-- find improper email addresses:
SELECT email
FROM   aminno_member_email
WHERE  email NOT REGEXP '^[^@]+@[^@]+\.[^@]{2,}$';

-- create a new column to denote if the email is valid
-- set the default to false
ALTER TABLE aminno_member_email
  ADD COLUMN is_valid_email_address BOOLEAN NOT NULL DEFAULT 0;

-- change the columns for ones with valid email addresses
UPDATE aminno_member_email
SET    is_valid_email_address = 1
WHERE  email REGEXP '^[^@]+@[^@]+\.[^@]{2,}$';

SELECT Count(*)
FROM   aminno_member_email
WHERE  is_valid_email_address = 1;

-- extract Top Level Domain
-- SUBSTR ( String, start index, number of characters to extract)
-- SUBSTRING_INDEX ()
-- Return a substring of a string before a specified number of delimiter occurs:
SELECT email_domain                           AS original,
       Substring_index(email_domain, '.', -2) AS top_level_domain
FROM   aminno_member_email
WHERE  is_valid_email_address = 1
LIMIT  30; 