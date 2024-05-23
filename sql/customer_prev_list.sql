-- This script is part of a larger library that I wrote at an endpoint 
-- protection company in the early 2010's to research customer complaints. 
-- Our customers' on-prem consoles uploaded a copy of their data every night to 
-- our data warehouse via a pipeline I built. We ran PostgreSQL queries on the DW 
-- to troubleshoot issues and research product ideas.
-- This collection of queries performs a few different tasks related to locating
--  the most prevalent Windows processes running, and prioritizing which ones 
--  needed to be human labeled.
-- Some customers complained that our product was incorrectly identifying 
-- important processes as extraneous, and was recommending that they prevent those 
-- important processes from automatically starting with Windows. If that had been 
-- done, then it would have likely caused major problems on thousands of laptops.
-- Other queries located which behavioral algorithm attributes (features) were 
-- contributing to incorrectly identifying the most prevalent Windows processes. 
-- Once we knew who the culprits were, we made adjustments to algorithm weights 
-- and fixed the issue.



-- get the number of unique agents that this specific customer has deployed
-- in the last 3 weeks
SELECT Count(DISTINCT clientguid)
FROM   digest_system
WHERE  customer = 'acme'
       AND Age(timestamp) < '21 day'; 


-- list of most prevalant processes for a specific customer
-- Requires that the prevalence column in label is updated recently (occurs on schedule every weekday morning)
DROP TABLE IF EXISTS tmp_cust_proc_prev;

SELECT di.filename,
       di.pathroot,
       di.probablecompanyname,
       Count(DISTINCT di.clientguid) AS customer_prevalence,
       lp.demeanor,
       lp.id                         AS label_id,
       lp.prevelance                 AS global_prevalence
INTO   tmp_cust_proc_prev
FROM   digest_image AS di
       JOIN label_process AS lp USING(filename, pathroot)
WHERE  di.filename LIKE '%.exe'
       AND di.customer = 'acme'
GROUP  BY di.filename,
          lp.demeanor,
          di.customer,
          lp.id,
          lp.description,
          lp.prevelance,
          di.pathroot,
          di.probablecompanyname
ORDER  BY Count(DISTINCT di.clientguid) DESC; 


-- display the most common processes that we haven't labeled by humans yet
-- these should be the priority for the next round of labeling
select * from tmp_cust_proc_prev
where
demeanor = 'NL';


-- create a new temporary table that only includes the autostarted processes
-- since those are the most important for certain use cases.
-- slow to run, needs indexing added
drop table if exists tmp_cust_proc_prev2;

SELECT   t1.filename,
         t1.pathroot,
         t1.probablecompanyname,
         t1.customer_prevalence,
         t1.demeanor,
         t1.label_id,
         t1.global_prevalence
INTO     tmp_cust_proc_prev2
FROM     tmp_cust_proc_prev AS t1
JOIN     digest_scoring     AS ds
using   (filename, pathroot)
WHERE    ds.extraneousattributes ilike '%autostart%'
AND      t1.customer_prevalence > 9
GROUP BY t1.filename,
         t1.pathroot,
         t1.probablecompanyname,
         t1.customer_prevalence,
         t1.demeanor,
         t1.label_id,
         t1.global_prevalence,
         ds.extraneousattributes
ORDER BY t1.customer_prevalence DESC;


-- do the same as above, but just list the autostarting ones that are not labeled
-- storing into a new table so it is persistent for a few days for coworkers to access
SELECT filename,
       pathroot,
       probablecompanyname,
       customer_prevalence,
       demeanor,
       label_id,
       global_prevalence
FROM   tmp_cust_proc_prev2
WHERE  demeanor = 'NL'
ORDER  BY filename; 


-- look for recent executables (exclude DLLs) that autostart
-- DLLs are often a lot more dangerous to prevent from autostarting
DROP TABLEIF EXISTS tmp_cust_proc_prev3;SELECT lp.pathroot,
       lp.filename,
       lp.id,
       lp.vendorname,
       lp.demeanor,
       lp.prevelance
INTO   tmp_cust_proc_prev3
FROM   label_process  AS lp
JOIN   digest_scoring AS ds
using (filename, pathroot)
WHERE  lp.filename LIKE '%.exe'
AND    Age(ds.timestamp) < '365 day'
AND    ds.extraneousattributes ilike '%autostart%'
AND    lp.prevelance >= 1;


-- get some counts:
select count (distinct id) from tmp_cust_proc_prev3;

select count (distinct(filename, pathroot) ) from digest_scoring;

select count (distinct(filename, pathroot) ) from digest_scoring
	where
	extraneousattributes ilike '%autostart%';
