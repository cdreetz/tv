import json
import asyncio
import re
import os
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
from playwright.async_api import async_playwright
import anthropic
from typing import Dict, List, Optional, AsyncGenerator, Any
from pydantic import BaseModel
from .common import PipelineStatus

load_dotenv()

class FounderProfile(BaseModel):
    """Profile of a company founder"""
    name: str
    past_companies: List[str] = []
    past_roles: List[str] = []
    universities: List[str] = []
    experience_summary: Optional[str] = None

class StartupProfile(BaseModel):
    """Profile of a startup company"""
    product_offering: Optional[str] = None
    differentiator: Optional[str] = None
    go_to_market_strategy: Optional[str] = None
    stage_round: Optional[str] = None
    total_funding: Optional[str] = None
    target_market: Optional[str] = None
    business_model: Optional[str] = None
    key_metrics: Optional[str] = None
    competitive_advantage: Optional[str] = None
    
    class Config:
        # Allow extra fields and convert types when possible
        extra = "allow"
        str_strip_whitespace = True

class CompanyExtraction(BaseModel):
    """Extracted data about a company"""
    founders: List[FounderProfile] = []
    startup_profile: StartupProfile
    market_category: List[str] = []
    technology_stack: List[str] = []

class MarketMap:
    """Class for managing market map operations"""
    def __init__(self, use_existing_login=False):
        self.data_folder = "8d7b3ce6f4596ddf83d6d955017a8210/"
        self.company_llm_summaries = self.data_folder + "company_llm_summaries.json"
        
        with open(self.company_llm_summaries, 'r') as f:
            content = json.load(f)
            self.companies = content
            self.companies_list = list(content.values())
            self.company_names = list(content.keys())
        
        self.linkedin_context = None
        self.use_existing_login = use_existing_login
        
        # Initialize Claude client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            print("Claude client initialized successfully")
        else:
            self.claude_client = None
            print("WARNING: ANTHROPIC_API_KEY not found in environment variables")

    async def extract_structured_data_stream(self, company_summary: str) -> AsyncGenerator[PipelineStatus, None]:
        """Use Claude API to extract structured data from company summary with streaming updates"""
        
        if not self.claude_client:
            yield PipelineStatus("error", "Claude client not initialized - check ANTHROPIC_API_KEY", 0.0, 
                               error="Claude client not initialized")
            return
        
        yield PipelineStatus("claude_extraction", "Preparing Claude API request...", 0.1)
        
        extraction_prompt = f"""Extract structured information from this company summary. Be precise and only extract information that is explicitly stated.

        Company Summary:
        {company_summary}

        Return ONLY valid JSON in this exact format:
        {{
            "founders": [
                {{
                    "name": "Full Name",
                    "past_companies": ["Company1", "Company2"],
                    "past_roles": ["Software Engineer @ Company1", "Product Manager @ Company2"],
                    "universities": ["University Name"],
                    "experience_summary": "Brief background summary"
                }}
            ],
            "startup_profile": {{
                "product_offering": "What the company builds/offers",
                "differentiator": "What makes them unique if mentioned",
                "go_to_market_strategy": "How they acquire customers if mentioned",
                "stage_round": "Funding stage mentioned if mentioned",
                "total_funding": "Amount raised if mentioned",
                "target_market": "Target customers if mentioned",
                "business_model": "Revenue model if mentioned",
                "key_metrics": "Single string with metrics like 'revenue: $4M, users: 100K' if mentioned",
                "competitive_advantage": "Main competitive edge if mentioned"
            }},
            "market_category": ["category1", "category2"],
            "technology_stack": ["tech1", "tech2"],
            "key_terms": ["term1", "term2"]
        }}

        Market Category options: 
        - Consumer vs B2B
        - Enterprise vs SMB
        - AI vs Non-AI
        - Data labeling
        - Infrastructure as a service
        - Agent based
            - Voice agents
            - Chatbots
            - Agentic analytics
        - If not listed above, come up with a category that fits

        Key Terms options:
        - brand
        - science
        - health
        - education
        - finance
        - gaming
        - media
        - enterprise
        - consumer
        - developer
        - analysis
        - outreach


        Extract ALL explicitly mentioned companies and roles. Return only the JSON, no other text."""

        try:
            yield PipelineStatus("claude_extraction", "Calling Claude API for structured extraction...", 0.3)
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": extraction_prompt}]
            )
            
            yield PipelineStatus("claude_extraction", "Received response from Claude API", 0.6)
            
            response_text = response.content[0].text
            
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                yield PipelineStatus("error", "No JSON found in Claude response", 0.0, 
                                   error="No JSON found in Claude response")
                return
            
            yield PipelineStatus("claude_extraction", "Parsing JSON response...", 0.8)
            json_str = response_text[json_start:json_end]
            
            try:
                extracted_data = json.loads(json_str)
                
                # Fix key_metrics if it's a dictionary instead of string
                if 'startup_profile' in extracted_data and 'key_metrics' in extracted_data['startup_profile']:
                    key_metrics = extracted_data['startup_profile']['key_metrics']
                    if isinstance(key_metrics, dict):
                        # Convert dictionary to formatted string
                        metrics_parts = []
                        for key, value in key_metrics.items():
                            metrics_parts.append(f"{key}: {value}")
                        extracted_data['startup_profile']['key_metrics'] = ", ".join(metrics_parts)
                
                result = CompanyExtraction(**extracted_data)
                yield PipelineStatus("claude_extraction", f"Structured data extraction successful: {len(result.founders)} founders found", 1.0, data=result.dict())
                
            except json.JSONDecodeError as e:
                yield PipelineStatus("error", f"JSON decode error: {e}", 0.0, error=f"JSON decode error: {e}")
                
        except Exception as e:
            yield PipelineStatus("error", f"Claude API error: {e}", 0.0, error=f"Claude API error: {e}")

    async def extract_structured_data(self, company_summary: str) -> Dict:
        """Use Claude API to extract structured data from company summary (non-streaming version)"""
        async for status in self.extract_structured_data_stream(company_summary):
            if status.stage == "claude_extraction" and status.data:
                return status.data
            elif status.stage == "error":
                print(f"Claude API error: {status.error}")
                return CompanyExtraction(startup_profile=StartupProfile()).dict()
        
        return CompanyExtraction(startup_profile=StartupProfile()).dict()

    async def enrich_company_data_stream(self, company_name: str) -> AsyncGenerator[PipelineStatus, None]:
        """Enrich company data with structured extraction with streaming updates"""
        if company_name not in self.companies:
            yield PipelineStatus("error", f"Company {company_name} not found in database", 0.0, 
                               error=f"Company {company_name} not found")
            return
        
        yield PipelineStatus("data_preparation", f"Loading company data for {company_name}", 0.1)
        
        company_data = self.companies[company_name].copy()
        summary = company_data['company_summary']
        
        yield PipelineStatus("data_preparation", f"Company summary loaded ({len(summary)} characters)", 0.2)
        
        # Stream the structured data extraction
        structured_data = None
        async for status in self.extract_structured_data_stream(summary):
            # Re-emit the status with adjusted progress (0.2 to 0.9)
            adjusted_progress = 0.2 + (status.progress * 0.7)
            yield PipelineStatus(status.stage, status.message, adjusted_progress, status.data, status.error)
            
            if status.stage == "claude_extraction" and status.data:
                structured_data = status.data
            elif status.stage == "error":
                return
        
        if structured_data:
            company_data['structured_data'] = structured_data
            yield PipelineStatus("completion", f"Data enrichment completed for {company_name}", 1.0, data=company_data)
        else:
            yield PipelineStatus("error", "Failed to extract structured data", 0.0, 
                               error="Failed to extract structured data")

    async def enrich_company_data(self, company_name: str) -> Dict:
        """Enrich company data with structured extraction (non-streaming version)"""
        async for status in self.enrich_company_data_stream(company_name):
            if status.stage == "completion":
                return status.data
            elif status.stage == "error":
                print(f"Error enriching {company_name}: {status.error}")
                return {}
        return {}

    async def get_social_links_from_url(self, url):
        """Extract social media links from a website"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            
            try:
                print(f"Loading {url}...")
                await page.goto(url, timeout=60000)
                await page.wait_for_timeout(3000)
                
                # Scroll to render all content
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                await page.wait_for_timeout(2000)
                await page.evaluate("window.scrollTo(0, 0);")
                await page.wait_for_timeout(2000)
                
                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except:
                    print("Network didn't go idle, proceeding anyway...")
                
                social_links = {}
                links = await page.query_selector_all('a[href]')
                print(f"Found {len(links)} links on the page")
                
                for link in links:
                    href = await link.get_attribute('href')
                    if href:
                        href_lower = href.lower()
                        if 'linkedin.com' in href_lower and 'linkedin' not in social_links:
                            social_links['linkedin'] = href
                            print(f"Found LinkedIn: {href}")
                        elif ('x.com' in href_lower or 'twitter.com' in href_lower) and 'twitter' not in social_links:
                            social_links['twitter'] = href
                            print(f"Found Twitter: {href}")
                
                await browser.close()
                return social_links
                
            except Exception as e:
                print(f"Error getting social links: {e}")
                await browser.close()
                return {}

    async def setup_linkedin_context(self):
        """Setup LinkedIn browser context"""
        if self.linkedin_context is None:
            self.playwright = await async_playwright().start()
            
            if self.use_existing_login:
                print("Using existing LinkedIn login...")
                profile_dir = "./browser_data"
            else:
                print("Starting fresh LinkedIn session...")
                profile_dir = "./browser_data_fresh"
                import shutil
                if os.path.exists(profile_dir):
                    shutil.rmtree(profile_dir)
            
            self.linkedin_context = await self.playwright.chromium.launch_persistent_context(
                profile_dir,
                headless=False
            )
        return self.linkedin_context

    async def close_linkedin_context(self):
        """Close LinkedIn browser context"""
        if self.linkedin_context:
            await self.linkedin_context.close()
            await self.playwright.stop()
            self.linkedin_context = None

    async def login_to_linkedin(self, page):
        """Login to LinkedIn using credentials from .env"""
        email = os.getenv('LINKEDIN_EMAIL')
        password = os.getenv('LINKEDIN_PASSWORD')
        
        if not email or not password:
            print("LinkedIn credentials not found in .env file")
            return False
        
        try:
            print(f"Attempting to log into LinkedIn with {email}...")
            
            email_input = await page.query_selector('input[name="session_key"]')
            if email_input:
                await page.fill('input[name="session_key"]', email)
                await page.wait_for_timeout(1000)
                
                await page.fill('input[name="session_password"]', password)
                await page.wait_for_timeout(1000)
                
                await page.click('button[type="submit"]')
                await page.wait_for_timeout(8000)
                
                current_url = page.url
                print(f"After login attempt, current URL: {current_url}")
                
                # Check if logged in
                profile_element = await page.query_selector('[data-control-name="identity_profile_photo"]')
                if profile_element or 'feed' in current_url or '/in/' in current_url:
                    print("Successfully logged into LinkedIn")
                    return True
                else:
                    print("Login may have failed or requires verification")
                    return False
            else:
                print("Email input not found - may already be logged in")
                return True
                
        except Exception as e:
            print(f"Error during LinkedIn login: {e}")
            return False

    def is_recent_post(self, time_text):
        """Check if post time indicates it's within the last month"""
        time_text = time_text.lower().strip()
        
        # Clean up the text
        time_text = re.sub(r'[•·\n\r]', ' ', time_text).strip()
        time_text = re.sub(r'\s+', ' ', time_text)
        
        # Extract time portion
        time_parts = time_text.split()
        if time_parts:
            time_portion = ' '.join(time_parts[:3])
            
            # Patterns for recent posts (within last month)
            recent_patterns = [
                r'(\d+)\s*m(?:in|ins|inute|inutes)?(?:\s|$)',  # minutes
                r'(\d+)\s*h(?:r|rs|our|ours)?(?:\s|$)',       # hours  
                r'(\d+)\s*d(?:ay|ays)?(?:\s|$)',              # days
                r'(\d+)\s*w(?:eek|eeks?)?(?:\s|$)',           # weeks
                r'(\d+)\s*mo(?:nth|nths?)?(?:\s|$)',          # months
            ]
            
            for pattern in recent_patterns:
                match = re.search(pattern, time_portion)
                if match:
                    number = int(match.group(1))
                    
                    if 'w' in time_portion:  # weeks
                        return number <= 4
                    elif 'd' in time_portion:  # days
                        return number <= 30
                    elif any(x in time_portion for x in ['h', 'hour']):  # hours
                        return True
                    elif any(x in time_portion for x in ['m', 'min']):  # minutes
                        return True
                    elif 'mo' in time_portion:  # months
                        return number <= 1
        
        return False

    async def count_recent_linkedin_posts(self, linkedin_url, context):
        """Count LinkedIn posts from the last month"""
        posts_url = linkedin_url.replace('/about/', '/posts/').replace('/company/', '/company/')
        if not posts_url.endswith('/posts/'):
            posts_url = posts_url.rstrip('/') + '/posts/'
        
        page = await context.new_page()
        
        try:
            print(f"Counting recent posts from: {posts_url}")
            await page.goto(posts_url, timeout=60000)
            await page.wait_for_timeout(5000)
            
            recent_post_count = 0
            scrolls = 0
            max_scrolls = 5
            processed_posts = set()
            
            while scrolls < max_scrolls:
                time_selectors = [
                    '[data-test-id="post-time"]',
                    'time',
                    '[aria-label*="ago"]',
                    '.feed-shared-actor__sub-description',
                    '.update-components-actor__sub-description',
                    '.feed-shared-text'
                ]
                
                found_times = []
                
                for selector in time_selectors:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        try:
                            time_text = await element.inner_text()
                            if not time_text:
                                time_text = await element.get_attribute('aria-label') or ''
                            
                            if time_text and time_text not in processed_posts:
                                found_times.append(time_text)
                                processed_posts.add(time_text)
                        except:
                            continue
                
                # Process found times
                for time_text in found_times:
                    if self.is_recent_post(time_text):
                        recent_post_count += 1
                        print(f"✓ Recent post found: '{time_text}'")
                
                # Scroll to load more posts
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                await page.wait_for_timeout(3000)
                scrolls += 1
                
                # Check if we've reached old posts
                page_text = await page.inner_text('body')
                if any(old_indicator in page_text.lower() for old_indicator in ['2mo', '3mo', '4mo', '2023', '2022']):
                    print("Reached older posts, stopping scroll")
                    break
            
            print(f"Total recent LinkedIn posts found: {recent_post_count}")
            await page.close()
            return recent_post_count
            
        except Exception as e:
            print(f"Error counting LinkedIn posts: {e}")
            await page.close()
            return 0

    async def get_linkedin_stats(self, linkedin_url):
        """Get LinkedIn follower count and recent posts"""
        context = await self.setup_linkedin_context()
        page = await context.new_page()
        
        try:
            if not self.use_existing_login:
                print("Going to LinkedIn login page...")
                await page.goto("https://www.linkedin.com/login", timeout=60000)
                await page.wait_for_timeout(3000)
                
                page_text = await page.inner_text('body')
                if 'Email' in page_text and 'Password' in page_text:
                    print("Logging in with new credentials...")
                    login_success = await self.login_to_linkedin(page)
                    if not login_success:
                        print("Login failed, skipping LinkedIn stats")
                        await page.close()
                        return {}
            else:
                print("Using existing LinkedIn session...")
                await page.goto("https://www.linkedin.com/feed", timeout=60000)
                await page.wait_for_timeout(3000)
            
            # Navigate to company page
            print(f"Navigating to: {linkedin_url}")
            await page.goto(linkedin_url, timeout=60000)
            await page.wait_for_timeout(5000)
            
            # Scroll to load content
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            await page.wait_for_timeout(2000)
            await page.evaluate("window.scrollTo(0, 0);")
            await page.wait_for_timeout(2000)
            
            stats = {}
            page_text = await page.inner_text('body')
            
            # Check if still on login page
            if 'Email' in page_text and 'Password' in page_text:
                print("Still on login page, LinkedIn access failed")
                await page.close()
                return {}
            
            # LinkedIn follower patterns
            follower_patterns = [
                r'(\d+(?:,\d+)*)\s+followers',
                r'(\d+(?:\.\d+)?[KkMm]?)\s+followers',
                r'followers[:\s]*(\d+(?:,\d+)*)',
                r'(\d+(?:,\d+)*)\s+follower',
                r'(\d+(?:,\d+)*)\s*•\s*\d+\s*employees',
                r'(\d+(?:,\d+)*)\s*followers\s*on\s*linkedin'
            ]
            
            for pattern in follower_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    stats['followers'] = match.group(1)
                    print(f"Found followers: {match.group(1)}")
                    break
            
            await page.close()
            
            # Count recent posts
            recent_posts = await self.count_recent_linkedin_posts(linkedin_url, context)
            stats['recent_posts'] = recent_posts
            
            print(f"LinkedIn stats found: {stats}")
            return stats
            
        except Exception as e:
            print(f"Error getting LinkedIn stats: {e}")
            await page.close()
            return {}

    async def get_twitter_stats(self, twitter_url):
        """Get Twitter follower count and posts"""
        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(
                "./twitter_data",
                headless=False
            )
            page = await context.new_page()
            
            try:
                print(f"Navigating to: {twitter_url}")
                await page.goto(twitter_url, timeout=60000)
                await page.wait_for_timeout(3000)
                
                # Scroll to load content
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                await page.wait_for_timeout(2000)
                await page.evaluate("window.scrollTo(0, 0);")
                await page.wait_for_timeout(2000)
                
                stats = {}
                page_text = await page.inner_text('body')
                
                # Twitter follower patterns
                follower_patterns = [
                    r'(\d+(?:,\d+)*)\s+followers',
                    r'(\d+(?:\.\d+)?[KkMm]?)\s+followers',
                    r'followers[:\s]*(\d+(?:,\d+)*)'
                ]
                
                for pattern in follower_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE)
                    if match:
                        stats['followers'] = match.group(1)
                        break
                
                # Look for posts count
                post_patterns = [
                    r'(\d+(?:,\d+)*)\s+posts',
                    r'(\d+(?:,\d+)*)\s+tweets',
                    r'posts[:\s]*(\d+(?:,\d+)*)',
                    r'tweets[:\s]*(\d+(?:,\d+)*)'
                ]
                
                for pattern in post_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE)
                    if match:
                        stats['posts'] = match.group(1)
                        break
                
                print(f"Twitter stats found: {stats}")
                
                await page.wait_for_timeout(2000 + int(await page.evaluate("Math.random() * 2000")))
                await context.close()
                return stats
                
            except Exception as e:
                print(f"Error getting Twitter stats: {e}")
                await context.close()
                return {}

async def main():
    parser = argparse.ArgumentParser(description='Market Map Social Media Scraper with Claude API')
    parser.add_argument('--existing-login', action='store_true', 
                        help='Use existing LinkedIn login instead of fresh credentials from .env')
    parser.add_argument('--company', type=str, default=None,
                        help='Specific company to analyze')
    parser.add_argument('--extract-only', action='store_true',
                        help='Only run structured extraction, skip social media scraping')
    args = parser.parse_args()
    
    mm = MarketMap(use_existing_login=args.existing_login)
    
    try:
        # Determine which company to analyze
        if args.company and args.company in mm.company_names:
            company_name = args.company
        else:
            company_name = mm.company_names[0]
        
        print(f"Analyzing: {company_name}")
        
        # Get enriched company data with Claude extraction
        enriched_data = await mm.enrich_company_data(company_name)
        print(f"Structured extraction complete for {company_name}")
        print(f"Extracted data: {json.dumps(enriched_data['structured_data'], indent=2)}")
        
        if not args.extract_only:
            # Get social media links from their website
            social_links = await mm.get_social_links_from_url("https://www.100ms.live/")
            print(f"Social Links: {social_links}")
            
            # Get social media stats
            social_stats = {}
            
            if 'linkedin' in social_links:
                print("Getting LinkedIn stats...")
                linkedin_stats = await mm.get_linkedin_stats(social_links['linkedin'])
                social_stats['linkedin'] = linkedin_stats
            
            if 'twitter' in social_links:
                print("Getting Twitter stats...")
                twitter_stats = await mm.get_twitter_stats(social_links['twitter'])
                social_stats['twitter'] = twitter_stats
            
            print(f"Final Social Stats: {social_stats}")
            
            # Add to company data
            enriched_data['social_links'] = social_links
            enriched_data['social_stats'] = social_stats
        
        print(f"Final enriched data: {json.dumps(enriched_data, indent=2, default=str)}")
        
    finally:
        await mm.close_linkedin_context()

if __name__ == "__main__":
    asyncio.run(main())