# [REDACTED] Project

## TODOs

- [x] Read through the summaries example
- [x] Read through the mm example
- [] Check out the emb map
- [x] Extract from current data
- [x] Extract startup name
- [x] Extract founder names
- [] Extract founder profiles (background, uni, etc)
- [] Once provided data has been exhausted time to scrape
- [] Find company URLs
- [] Find company socials from URL (twitter, linkedin)
- [] Gather socials stats (followers, recent posts)
- [] Crunchbase data, past funding (apparently also might be on their linkedin)

## Notes

- I think a potentially good way to make navigating the map is through a hierarchy
- Run all summaries through simple classification steps to create labels

  - Ex
  - AI vs Non AI (although could probably assume all AI lol)
  - Consumer vs SaaS
  - SaaS
  - Processing stuff
    - Information processing at scale
    - Financial research automation
    - Data labeling
  - Generating stuff
    - Marketing, social, etc
  - Assistants
    - Back office
      - Medical
      - Other
    - Personal

#### 2hr mark
- So far have the basis for extracting the given info from summaries
- Also have some starting enrichment processes for scraping from company sites, linkedin, and twitter
- Going back to read the challenge again for like the 7th time so i dont lose the plot
- 'organize companies into those which have similar offerings, products, or market positions'
- Google says 'market position' is 'Market positioning is a strategic exercise we use to establish the image of a brand or product in a consumer's mind. This is achieved through the four Ps: promotion, price, place, and product'
- The summaries kind of already provide the ability to find similar offerings or products

#### 2.5hr
- lightbulb, initially i was opposed to using the summaries provided to do emb stuff bc reading alot of the summaries a good amount of the summaries have mostly information on the founders and little information on the product.
- so just reprocess the company summaries to extract product summaries. hard part here is what to do with company summaries that basically have nothing on the product itself
- wow after reading even more of the summaries, they kinda suck. most of the ones ive read are mostly just founder notes with little to no info on product


#### 3hr
- okay went back to focus on the actual focus of organizing and comparing similar startups
- report generator has 2 main functions, first takes a single startup as input and gathers market position and general profile


#### 3.5hr 
- after reading the problem statement for the 10th time have i had an epiphany. when they say 'the number one goal is to develop a way for us to create these market maps moving forward' i should probably focus on developing a way to create these market maps moving forward..... idk why im creating reports 
- luckily the 3d umap projections are already provided, so lets assume the user wants to look at the 3d market map, lets work on adding some enrichments to this directly
- ie while viewing the map, filter for round, experienced founders, key word search, etc but its all over the same map

#### 4.5hr 
- okay i am making an executive decision to scrap the 3d map stuff, just not a very intuitive way to find startups and more so with the fact that summaries are low quality for this task specifically and reprompting to generate new summaries is cost inhibitive 
- going back to report generator thing
- if i put myself in the shoes of an investor, do i want to try to navigate this weird map with a bunch of dots? hell no.  do i want to give some tool a single startup or even a list of startups and in return get a comprehensive overview of the startups, the founders, their market placements, their go to market strategies, comaprisons with similar startups?  hell yeah. (i think)

#### Failures

- ~repeatedly spending time on what i thought would be useful for investors and not the single goal stated, creating market maps moving forward~
- after further consideration the above statement was in fact a feature not a bug
- mapped 3d projectionsk
- used example market map to color different groups on the map
- lots of low quality summaries, gonna rewrite summaries to actual product/offering summaries
