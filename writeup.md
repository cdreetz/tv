# ThVe

## Introduction

Decision makers primary role is to consume information and make yes or no decisions that have big impacts.  The reason these people are paid highly is for no reason besides the fact that they are trusted to make these decisions in ways that result in positive impact.  Entire teams are formed around the fact that there is so much information decision makers need to consider that it is impossible for them as an individual to find, aggregate, navigate, and consume over such large information landscapes in a reasonable time and effort.

Breaking down these roles there are 2 primary tasks, consume information and output a decision.  Which of these is the most time consuming? The consumption.  Which one is best left to a trusted human to perform?  The output decision.  This is why LLMs that can consume almost endless amounts of information from nearly endless sources, aggregate that information, and output a more reasonable subset of information for a decision maker to quickly consume and make a decision, is a match made in heaven.


## My process

### Phase 1

Just as the challenge stated, I read the challenge a few times over the course of a couple days before even actually starting it.  Imagining different scenarios of what I could create to best help investors navigate these market maps.  

Once I started my initial thought process was the given data is not enough to create an informative and useful market map, so I should create some things that can scrape for additional data.  Before reading enough of the company summaries provided, I had made an incorrect assumption that the summaries were primarily about the companies and products, so I should make something that helps build better founder profiles.  Starting by extracting founder names from the summaries, then pairing founder names and any additional information, we search for them on LinkedIn and try to verify their identity with the combination of information so we try to not get the wrong person.  

This was followed by trying to gather information on the companies social presence, so assuming we have the companies URL, we scrape their landing page for socials links to get their Linked and Twitter links.  From there we then scrape their twitter for number of followers and number of posts to see how active they are.  Same thing for linkedin but in this case we have to load their posts page, scroll down and try to count how many posts they have within the last month.

### Phase 2

This doesnt really scale well for an initial demo for a few reasons. First of all I was using my personal accounts to scrape and didn't want to get banned so I set really long wait times which made the scraping slow.  Also just for the first company, there was a weird scenario because I believe they had pivoted or something because the website linked to socials that linked to a different website, so it was honeslty unclear what their actual site was which would potentially result in incorrectly scraped information.  With that I moved on.

Going back to read I wanted to focus on the 'map' part with the 3d projections, eventually realizing the summaries were kinda bad as far as a good portion of them had little to no infornation on product/offering which made them bad for building market maps.  

### Phase 3

Eventually I returned back to my original pipeline and report generator, go it clean up, and converted to a streaming API.  Whipped up a UI and it is now a decent looking applicaiton where an investor can simply input a list of companies they want to compare, start a job, a bunch of asyncronous things happen including extracting data from summaries, compiling it, and writing it out to a clear single page comparison of all the companies, founders, differences, and recommendations.


### Delivery

![Screenshot of the application](Screenshot%202025-05-27%20at%204.55.24%20PM.png)
