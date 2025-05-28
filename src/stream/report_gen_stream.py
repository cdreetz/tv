import json
import asyncio
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .big_scraper_stream import MarketMap
from .common import PipelineStatus
from dotenv import load_dotenv
from typing import Dict, List, AsyncGenerator, Any
import re

load_dotenv()

class MarketReportGenerator:
    def __init__(self):
        self.market_map = MarketMap()
        self.companies = self.market_map.companies
        self.company_names = list(self.companies.keys())
        
        # Precompute similarity matrix
        self._compute_similarity_matrix()
    
    def _compute_similarity_matrix(self):
        """Compute TF-IDF similarity matrix for all companies"""
        summaries = [company['company_summary'] for company in self.companies.values()]
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(summaries)
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        self.vectorizer = vectorizer
    
    def find_similar_companies(self, company_name, market_categories=None, key_terms=None, search_terms=None, product_offering_short=None, top_k=5):
        """Find most similar companies using enhanced matching with product offerings and search terms"""
        if company_name not in self.company_names:
            return []
        
        # Get the original company's data for comparison
        original_company_data = self.companies[company_name]
        original_structured_data = original_company_data.get('structured_data', {})
        original_startup_profile = original_structured_data.get('startup_profile', {})
        
        # Build comprehensive search terms list
        all_search_terms = []
        
        # Add explicit search terms (highest priority)
        if search_terms:
            all_search_terms.extend([term.lower().strip() for term in search_terms])
        
        # Add product offering short terms (high priority for semantic matching)
        product_terms = []
        if product_offering_short:
            product_terms = [term.lower().strip() for term in product_offering_short.split() if len(term) > 2]
            all_search_terms.extend(product_terms)
        
        # Add market categories and key terms (medium priority)
        if market_categories:
            all_search_terms.extend([term.lower().strip() for term in market_categories])
        if key_terms:
            all_search_terms.extend([term.lower().strip() for term in key_terms])
        
        # If no search terms available, fall back to TF-IDF similarity
        if not all_search_terms:
            company_idx = self.company_names.index(company_name)
            similarities = self.similarity_matrix[company_idx]
            similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            similar_companies = []
            for idx in similar_indices:
                similar_company_name = self.company_names[idx]
                similar_company_data = self.companies[similar_company_name]
                similar_structured_data = similar_company_data.get('structured_data', {})
                similar_startup_profile = similar_structured_data.get('startup_profile', {})
                
                similar_companies.append({
                    'name': similar_company_name,
                    'similarity_score': similarities[idx],
                    'summary': similar_company_data['company_summary'],
                    'similarity_reason': 'Content similarity based on company descriptions',
                    'key_differentiator': similar_startup_profile.get('differentiator') or similar_startup_profile.get('competitive_advantage') or 'Differentiation not clearly specified'
                })
            return similar_companies
        
        print(f"Enhanced search terms: {all_search_terms}")  # Debug output
        print(f"Product offering short: {product_offering_short}")  # Debug output
        
        similar_companies = []
        
        # Search through all companies
        for other_company_name, company_data in self.companies.items():
            if other_company_name == company_name:
                continue  # Skip the company itself
            
            # Get structured data for comparison
            other_structured_data = company_data.get('structured_data', {})
            other_startup_profile = other_structured_data.get('startup_profile', {})
            other_product_offering_short = other_startup_profile.get('product_offering_short', '')
            other_search_terms = other_structured_data.get('search_terms', [])
            
            summary = company_data['company_summary'].lower()
            
            # Enhanced scoring system
            total_score = 0
            matched_terms = []
            match_details = []
            
            # 1. Direct search terms matching (weight: 3.0)
            search_term_matches = 0
            for term in (search_terms or []):
                term_lower = term.lower().strip()
                pattern = r'\b' + re.escape(term_lower) + r'\b'
                
                if re.search(pattern, summary, re.IGNORECASE):
                    search_term_matches += 1
                    matched_terms.append(term)
                    match_details.append(f"search term: {term}")
                    print(f"Found search term '{term}' in {other_company_name}")
                
                # Also check in other company's search terms
                if other_search_terms and any(re.search(pattern, other_term, re.IGNORECASE) for other_term in other_search_terms):
                    search_term_matches += 0.5  # Partial match
                    if term not in matched_terms:
                        matched_terms.append(term)
                        match_details.append(f"search term match: {term}")
            
            total_score += search_term_matches * 3.0
            
            # 2. Product offering similarity (weight: 2.5)
            product_similarity_score = 0
            if product_offering_short and other_product_offering_short:
                # Direct product offering comparison
                if product_offering_short.lower() == other_product_offering_short.lower():
                    product_similarity_score = 1.0
                    match_details.append(f"exact product match: {product_offering_short}")
                else:
                    # Check for partial matches in product offerings
                    product_words = set(product_offering_short.lower().split())
                    other_product_words = set(other_product_offering_short.lower().split())
                    
                    # Remove common words
                    common_words = {'the', 'and', 'or', 'for', 'with', 'in', 'on', 'at', 'to', 'a', 'an'}
                    product_words = product_words - common_words
                    other_product_words = other_product_words - common_words
                    
                    if product_words and other_product_words:
                        overlap = len(product_words.intersection(other_product_words))
                        union = len(product_words.union(other_product_words))
                        if union > 0:
                            jaccard_similarity = overlap / union
                            if jaccard_similarity > 0.3:  # Threshold for meaningful similarity
                                product_similarity_score = jaccard_similarity
                                match_details.append(f"product similarity: {jaccard_similarity:.2f}")
                
                # Also check if product offering short appears in the other company's summary
                if product_offering_short:
                    for word in product_offering_short.lower().split():
                        if len(word) > 3:  # Skip short words
                            pattern = r'\b' + re.escape(word) + r'\b'
                            if re.search(pattern, summary, re.IGNORECASE):
                                product_similarity_score += 0.2
                                if word not in [m.split(': ')[1] if ': ' in m else m for m in matched_terms]:
                                    matched_terms.append(word)
                                    match_details.append(f"product term: {word}")
            
            total_score += product_similarity_score * 2.5
            
            # 3. Market categories and key terms matching (weight: 1.5)
            category_term_matches = 0
            for term in (market_categories or []) + (key_terms or []):
                term_lower = term.lower().strip()
                pattern = r'\b' + re.escape(term_lower) + r'\b'
                
                if re.search(pattern, summary, re.IGNORECASE):
                    category_term_matches += 1
                    if term not in matched_terms:
                        matched_terms.append(term)
                        match_details.append(f"category/key term: {term}")
            
            total_score += category_term_matches * 1.5
            
            # 4. Cross-reference with other company's search terms (weight: 2.0)
            if other_search_terms:
                cross_matches = 0
                for our_term in all_search_terms:
                    for their_term in other_search_terms:
                        # Check for exact or partial matches
                        if our_term == their_term.lower().strip():
                            cross_matches += 1
                            if our_term not in matched_terms:
                                matched_terms.append(our_term)
                                match_details.append(f"cross-reference: {our_term}")
                        elif len(our_term) > 3 and len(their_term) > 3:
                            # Check for substring matches
                            if our_term in their_term.lower() or their_term.lower() in our_term:
                                cross_matches += 0.5
                                if our_term not in matched_terms:
                                    matched_terms.append(our_term)
                                    match_details.append(f"partial cross-reference: {our_term}")
                
                total_score += cross_matches * 2.0
            
            # Calculate final similarity score (normalize by maximum possible score)
            max_possible_score = len(search_terms or []) * 3.0 + 2.5 + len((market_categories or []) + (key_terms or [])) * 1.5 + len(all_search_terms) * 2.0
            similarity_score = total_score / max_possible_score if max_possible_score > 0 else 0
            
            # Only include companies with meaningful matches
            if total_score > 0.5:  # Minimum threshold
                # Get key differentiator
                key_differentiator = other_startup_profile.get('differentiator') or other_startup_profile.get('competitive_advantage')
                if not key_differentiator:
                    key_differentiator = self._extract_brief_differentiator(company_data['company_summary'], other_company_name)
                
                # Generate enhanced similarity reason
                similarity_reason = self._generate_enhanced_similarity_reason(
                    matched_terms, match_details, product_offering_short, other_product_offering_short
                )
                
                similar_companies.append({
                    'name': other_company_name,
                    'similarity_score': min(similarity_score, 1.0),  # Cap at 1.0
                    'raw_score': total_score,
                    'summary': company_data['company_summary'],
                    'matched_terms': matched_terms,
                    'match_count': len(matched_terms),
                    'match_details': match_details,
                    'similarity_reason': similarity_reason,
                    'key_differentiator': key_differentiator,
                    'other_product_offering': other_product_offering_short
                })
        
        print(f"Found {len(similar_companies)} companies with enhanced matching")  # Debug output
        
        # Sort by raw score first, then by similarity score
        similar_companies.sort(key=lambda x: (x['raw_score'], x['similarity_score']), reverse=True)
        
        # Return top_k results
        return similar_companies[:top_k]
    
    def _generate_enhanced_similarity_reason(self, matched_terms, match_details, product_offering_short, other_product_offering_short):
        """Generate enhanced similarity reason based on match details"""
        if not match_details:
            return f"Matches on: {', '.join(matched_terms)}"
        
        # Group match details by type
        search_matches = [d for d in match_details if 'search term' in d]
        product_matches = [d for d in match_details if 'product' in d]
        category_matches = [d for d in match_details if 'category' in d or 'key term' in d]
        cross_matches = [d for d in match_details if 'cross-reference' in d]
        
        reasons = []
        
        if product_matches and product_offering_short and other_product_offering_short:
            if product_offering_short.lower() == other_product_offering_short.lower():
                reasons.append(f"identical product focus ({product_offering_short})")
            else:
                reasons.append(f"similar product offerings ({product_offering_short} ↔ {other_product_offering_short})")
        elif product_matches:
            reasons.append("related product features")
        
        if search_matches:
            reasons.append("direct search term alignment")
        
        if cross_matches:
            reasons.append("complementary market positioning")
        
        if category_matches:
            reasons.append("shared market categories")
        
        if not reasons:
            reasons.append(f"keyword overlap: {', '.join(matched_terms[:3])}")
        
        return "Both companies " + " and ".join(reasons)
    
    def _extract_brief_differentiator(self, summary, company_name):
        """Extract a brief differentiator from company summary"""
        # Look for common differentiator patterns
        summary_lower = summary.lower()
        
        # Patterns that often indicate differentiators
        differentiator_patterns = [
            r'unlike.*?[,.]',
            r'different.*?[,.]',
            r'unique.*?[,.]',
            r'first.*?[,.]',
            r'only.*?[,.]',
            r'proprietary.*?[,.]',
            r'innovative.*?[,.]',
            r'breakthrough.*?[,.]'
        ]
        
        for pattern in differentiator_patterns:
            match = re.search(pattern, summary_lower)
            if match:
                differentiator = match.group(0).strip(' ,.')
                if len(differentiator) < 100:  # Keep it brief
                    return differentiator.capitalize()
        
        # Fallback: extract first sentence that mentions the company or a key feature
        sentences = summary.split('.')
        for sentence in sentences[:3]:  # Check first 3 sentences
            if len(sentence.strip()) > 20 and len(sentence.strip()) < 150:
                return sentence.strip()
        
        return "Specific differentiation not clearly specified"

    async def _generate_claude_similarity_analysis(self, company_name: str, company_summary: str, similar_company_name: str, similar_company_summary: str) -> Dict[str, str]:
        """Use Claude to generate sophisticated similarity and differentiator analysis"""
        if not self.market_map.claude_client:
            return {
                'similarity_reason': f"Both companies operate in similar market spaces",
                'key_differentiator': "Differentiation analysis not available - Claude client not initialized"
            }
        
        analysis_prompt = f"""Analyze these two companies and provide insights on their similarities and key differences.

Company A: {company_name}
Description: {company_summary}

Company B: {similar_company_name}  
Description: {similar_company_summary}

Please provide a JSON response with exactly these two fields:

{{
    "similarity_reason": "A 1-2 sentence explanation of how these companies are similar, focusing on their core business models, target markets, or product approaches. Start with 'Both companies...'",
    "key_differentiator": "A 1-2 sentence explanation of what makes {similar_company_name} unique or different from {company_name}, focusing on their specific competitive advantages, technology, or market positioning."
}}

Be concise, specific, and focus on the most important business-level similarities and differences. Avoid generic statements."""

        try:
            response = self.market_map.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                try:
                    analysis = json.loads(json_str)
                    return {
                        'similarity_reason': analysis.get('similarity_reason', f"Both companies operate in similar market spaces"),
                        'key_differentiator': analysis.get('key_differentiator', f"Specific differentiation for {similar_company_name} not clearly identified")
                    }
                except json.JSONDecodeError:
                    pass
            
            # Fallback if JSON parsing fails
            return {
                'similarity_reason': f"Both companies operate in similar market spaces",
                'key_differentiator': f"Specific differentiation for {similar_company_name} not clearly identified"
            }
            
        except Exception as e:
            print(f"Claude analysis error: {e}")
            return {
                'similarity_reason': f"Both companies operate in similar market spaces",
                'key_differentiator': f"Specific differentiation for {similar_company_name} not clearly identified"
            }

    async def generate_single_company_report_stream(self, company_name: str) -> AsyncGenerator[PipelineStatus, None]:
        """Generate comprehensive report for a single company with streaming updates"""
        try:
            yield PipelineStatus("initialization", f"Starting analysis for {company_name}", 0.0)
            
            if company_name not in self.companies:
                yield PipelineStatus("error", f"Company '{company_name}' not found in database", 0.0, 
                                   error=f"Company '{company_name}' not found")
                return
            
            yield PipelineStatus("validation", f"Company {company_name} found in database", 0.1)
            
            # Enrich the company data with structured extraction using streaming
            enriched_data = None
            async for status in self.market_map.enrich_company_data_stream(company_name):
                # Re-emit the status with adjusted progress (0.1 to 0.4)
                adjusted_progress = 0.1 + (status.progress * 0.3)
                
                if status.stage == "completion":
                    # Change the stage name so UI doesn't stop here
                    yield PipelineStatus("data_enrichment_complete", status.message, adjusted_progress, status.data, status.error)
                    enriched_data = status.data
                elif status.stage == "error":
                    yield PipelineStatus(status.stage, status.message, adjusted_progress, status.data, status.error)
                    return
                else:
                    yield PipelineStatus(status.stage, status.message, adjusted_progress, status.data, status.error)
            
            if not enriched_data:
                yield PipelineStatus("error", "Failed to enrich company data", 0.0, error="Data enrichment failed")
                return
            
            # Extract structured data
            yield PipelineStatus("data_extraction", "Extracting structured data...", 0.5)
            structured_data = enriched_data.get('structured_data', {})
            founders_info = structured_data.get('founders', [])
            startup_profile = structured_data.get('startup_profile', {})
            market_categories = structured_data.get('market_category', [])
            key_terms = structured_data.get('key_terms', [])
            search_terms = structured_data.get('search_terms', [])
            product_offering_short = startup_profile.get('product_offering_short', '')
            
            print(f"Extracted market_categories: {market_categories}")  # Debug output
            print(f"Extracted key_terms: {key_terms}")  # Debug output
            print(f"Extracted search_terms: {search_terms}")  # Debug output
            print(f"Extracted product_offering_short: {product_offering_short}")  # Debug output
            
            yield PipelineStatus("data_extraction", f"Found {len(founders_info)} founders and {len(market_categories)} market categories", 0.6)
            
            # Find similar companies using enhanced matching
            yield PipelineStatus("similarity_analysis", "Finding similar companies using enhanced matching with product offerings and search terms...", 0.7)
            similar_companies = self.find_similar_companies(
                company_name, 
                market_categories=market_categories, 
                key_terms=key_terms,
                search_terms=search_terms,
                product_offering_short=product_offering_short,
                top_k=4
            )
            
            print(f"Similar companies found: {len(similar_companies)}")  # Debug output
            for comp in similar_companies:
                print(f"  - {comp['name']}: {comp['similarity_score']:.3f} (raw: {comp.get('raw_score', 0):.2f}, matched: {comp.get('matched_terms', [])})")  # Debug output
            
            # Update the message based on what we found
            if similar_companies and 'match_details' in similar_companies[0]:
                yield PipelineStatus("similarity_analysis", f"Found {len(similar_companies)} similar companies using enhanced matching (product offerings + search terms)", 0.8)
            elif similar_companies and 'matched_terms' in similar_companies[0]:
                yield PipelineStatus("similarity_analysis", f"Found {len(similar_companies)} similar companies using keyword matching", 0.8)
            else:
                yield PipelineStatus("similarity_analysis", f"Found {len(similar_companies)} similar companies using TF-IDF similarity", 0.8)
            
            # Enhance similar companies with Claude analysis
            if similar_companies and self.market_map.claude_client:
                yield PipelineStatus("claude_analysis", "Enhancing similar companies with Claude-powered analysis...", 0.85)
                
                company_summary = enriched_data['company_summary']
                
                for i, similar_company in enumerate(similar_companies):
                    similar_company_name = similar_company['name']
                    similar_company_summary = similar_company['summary']
                    
                    # Generate Claude analysis for this similar company
                    claude_analysis = await self._generate_claude_similarity_analysis(
                        company_name, company_summary, similar_company_name, similar_company_summary
                    )
                    
                    # Update the similar company with Claude-generated insights
                    similar_company['claude_similarity_reason'] = claude_analysis['similarity_reason']
                    similar_company['claude_key_differentiator'] = claude_analysis['key_differentiator']
                    
                    # Progress update for each company analyzed
                    progress = 0.85 + (i + 1) / len(similar_companies) * 0.04  # 0.85 to 0.89
                    yield PipelineStatus("claude_analysis", f"Analyzed {similar_company_name} ({i+1}/{len(similar_companies)})", progress)
                
                yield PipelineStatus("claude_analysis", f"Claude analysis completed for {len(similar_companies)} similar companies", 0.89)
            
            # Get social links if available
            social_links = enriched_data.get('social_links', {})
            
            # Generate report
            yield PipelineStatus("report_generation", "Generating markdown report...", 0.9)
            
            report = f"""# Market Analysis Report: {company_name}

## Company Overview
{enriched_data['company_summary']}

## Founder Profiles
"""
            
            if founders_info:
                for founder in founders_info:
                    report += f"""
### {founder.get('name', 'Unknown')}
"""
                    if founder.get('past_companies'):
                        report += f"**Previous Companies:** {', '.join(founder['past_companies'])}\n"
                    if founder.get('past_roles'):
                        report += f"**Past Roles:** {', '.join(founder['past_roles'])}\n"
                    if founder.get('universities'):
                        report += f"**Education:** {', '.join(founder['universities'])}\n"
                    if founder.get('experience_summary'):
                        report += f"**Background:** {founder['experience_summary']}\n"
                    report += "\n"
            else:
                report += "Founder information not clearly identified in available data.\n"
            
            report += f"""
## Startup Profile
"""
            if startup_profile.get('product_offering'):
                report += f"**Product/Service:** {startup_profile['product_offering']}\n\n"
            if startup_profile.get('product_offering_short'):
                report += f"**Product Focus:** {startup_profile['product_offering_short']}\n\n"
            if startup_profile.get('differentiator'):
                report += f"**Key Differentiator:** {startup_profile['differentiator']}\n\n"
            if startup_profile.get('target_market'):
                report += f"**Target Market:** {startup_profile['target_market']}\n\n"
            if startup_profile.get('business_model'):
                report += f"**Business Model:** {startup_profile['business_model']}\n\n"
            if startup_profile.get('go_to_market_strategy'):
                report += f"**Go-to-Market:** {startup_profile['go_to_market_strategy']}\n\n"
            if startup_profile.get('stage_round'):
                report += f"**Funding Stage:** {startup_profile['stage_round']}\n\n"
            if startup_profile.get('total_funding'):
                report += f"**Total Funding:** {startup_profile['total_funding']}\n\n"
            if startup_profile.get('key_metrics'):
                report += f"**Key Metrics:** {startup_profile['key_metrics']}\n\n"
            
            report += f"""
## Market Positioning
**Market Categories:** {', '.join(market_categories) if market_categories else 'General technology'}
**Key Terms:** {', '.join(key_terms) if key_terms else 'Not specified'}
**Search Terms:** {', '.join(search_terms) if search_terms else 'Not specified'}

## Competitive Landscape
The following companies operate in similar spaces:

"""
            
            for i, similar in enumerate(similar_companies, 1):
                report += f"""
### {i}. {similar['name']} 
**Similarity Score:** {similar['similarity_score']:.3f}
"""
                # Show raw score if available (from enhanced matching)
                if 'raw_score' in similar:
                    report += f"**Raw Match Score:** {similar['raw_score']:.2f}\n"
                
                # Show matched terms if available (from keyword matching)
                if 'matched_terms' in similar and similar['matched_terms']:
                    report += f"**Matched Terms:** {', '.join(similar['matched_terms'])}\n"
                
                # Show other company's product offering if available
                if 'other_product_offering' in similar and similar['other_product_offering']:
                    report += f"**Their Product Focus:** {similar['other_product_offering']}\n"
                
                # Prioritize Claude-generated similarity reason over basic fallback
                if 'claude_similarity_reason' in similar and similar['claude_similarity_reason']:
                    report += f"**How They're Similar:** {similar['claude_similarity_reason']}\n"
                elif 'similarity_reason' in similar and similar['similarity_reason']:
                    report += f"**Similarity Reason:** {similar['similarity_reason']}\n"
                
                # Prioritize Claude-generated key differentiator over basic fallback
                if 'claude_key_differentiator' in similar and similar['claude_key_differentiator']:
                    report += f"**Key Differentiator:** {similar['claude_key_differentiator']}\n"
                elif 'key_differentiator' in similar and similar['key_differentiator']:
                    report += f"**Key Differentiator:** {similar['key_differentiator']}\n"
                
                report += f"**Brief:** {similar['summary'][:200]}...\n\n"
            
            report += f"""
## Social Media Presence
"""
            if social_links:
                for platform, url in social_links.items():
                    report += f"**{platform.title()}:** {url}\n"
            else:
                report += "Social media links not yet analyzed.\n"
            
            # Investment considerations using structured data
            competitive_advantage = startup_profile.get('competitive_advantage', '')
            has_clear_differentiator = bool(startup_profile.get('differentiator'))
            has_funding_info = bool(startup_profile.get('total_funding') or startup_profile.get('stage_round'))
            experienced_founders = len([f for f in founders_info if f.get('past_companies')]) > 0
            
            report += f"""
## Investment Considerations
- **Market Position:** {'Well-differentiated' if has_clear_differentiator else 'Moderate differentiation'} in {', '.join(market_categories[:2]) if market_categories else 'technology'} space
- **Competition Level:** {'High' if len(similar_companies) > 0 and similar_companies[0]['similarity_score'] > 0.3 else 'Moderate'} - {len(similar_companies)} similar companies identified
- **Founder Experience:** {'Strong' if experienced_founders else 'Limited information available'} - {len([f for f in founders_info if f.get('past_companies')])} founders with notable backgrounds
- **Funding Status:** {'Disclosed' if has_funding_info else 'Not available'}
- **Competitive Advantage:** {competitive_advantage if competitive_advantage else 'Not clearly articulated'}

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
            
            yield PipelineStatus("completion", f"Report generation completed for {company_name}", 1.0, data={
                'report': report,
                'similar_companies': similar_companies,
                'company_summary': enriched_data['company_summary'],
                'market_categories': market_categories,
                'key_terms': key_terms,
                'search_terms': search_terms,
                'product_offering_short': product_offering_short,
                'structured_data': structured_data
            })
            
        except Exception as e:
            yield PipelineStatus("error", f"Error generating report for {company_name}: {str(e)}", 0.0, error=str(e))

    async def generate_comparison_report_stream(self, company_names_list: List[str]) -> AsyncGenerator[PipelineStatus, None]:
        """Generate comparison report for multiple companies with streaming updates"""
        try:
            yield PipelineStatus("initialization", f"Starting comparison analysis for {len(company_names_list)} companies", 0.0)
            
            valid_companies = [name for name in company_names_list if name in self.companies]
            
            if not valid_companies:
                yield PipelineStatus("error", "No valid companies found in the provided list", 0.0, 
                                   error="No valid companies found")
                return
            
            yield PipelineStatus("validation", f"Found {len(valid_companies)} valid companies: {', '.join(valid_companies)}", 0.05)
            
            # Enrich all companies with structured data using streaming
            yield PipelineStatus("data_enrichment", "Starting data enrichment for all companies...", 0.1)
            enriched_companies = {}
            
            for i, company_name in enumerate(valid_companies):
                base_progress = 0.1 + (i / len(valid_companies)) * 0.4  # 0.1 to 0.5
                
                yield PipelineStatus("data_enrichment", f"Enriching data for {company_name}... ({i+1}/{len(valid_companies)})", base_progress)
                
                # Stream the enrichment for this company
                async for status in self.market_map.enrich_company_data_stream(company_name):
                    # Adjust progress within this company's allocation
                    company_progress = base_progress + (status.progress * (0.4 / len(valid_companies)))
                    
                    if status.stage == "completion":
                        # Change the stage name so UI doesn't stop here
                        yield PipelineStatus("data_enrichment_complete", f"{company_name}: {status.message}", company_progress, status.data, status.error)
                        enriched_companies[company_name] = status.data
                    elif status.stage == "error":
                        yield PipelineStatus("error", f"Failed to enrich {company_name}: {status.error}", 0.0, error=status.error)
                        return
                    else:
                        yield PipelineStatus(status.stage, f"{company_name}: {status.message}", company_progress, status.data, status.error)
            
            yield PipelineStatus("data_enrichment", "Data enrichment completed for all companies", 0.5)
            
            # Extract structured data for all companies
            yield PipelineStatus("data_extraction", "Extracting structured data for all companies...", 0.55)
            all_founders = {}
            all_startup_profiles = {}
            all_market_categories = {}
            
            for company_name in valid_companies:
                enriched_data = enriched_companies[company_name]
                structured_data = enriched_data.get('structured_data', {})
                
                all_founders[company_name] = structured_data.get('founders', [])
                all_startup_profiles[company_name] = structured_data.get('startup_profile', {})
                all_market_categories[company_name] = structured_data.get('market_category', [])
            
            yield PipelineStatus("data_extraction", "Structured data extraction completed", 0.6)
            
            # Generate comparison report
            yield PipelineStatus("report_generation", "Generating comparison report...", 0.7)
            
            report = f"""# Competitive Analysis Report

## Companies Analyzed
{', '.join(valid_companies)}

## Founder Comparison

"""
            
            for company_name in valid_companies:
                founders = all_founders[company_name]
                report += f"""
### {company_name}
"""
                if founders:
                    for founder in founders:
                        name = founder.get('name', 'Unknown')
                        past_companies = founder.get('past_companies', [])
                        past_roles = founder.get('past_roles', [])
                        
                        report += f"**{name}**\n"
                        if past_companies:
                            report += f"  - Previous: {', '.join(past_companies)}\n"
                        if past_roles:
                            report += f"  - Roles: {', '.join(past_roles)}\n"
                else:
                    report += "Founders not clearly identified\n"
            
            yield PipelineStatus("report_generation", "Generated founder comparison section", 0.75)
            
            report += f"""

## Product & Strategy Comparison

"""
            
            for company_name in valid_companies:
                startup_profile = all_startup_profiles[company_name]
                report += f"""
### {company_name}
"""
                if startup_profile.get('product_offering'):
                    report += f"**Product:** {startup_profile['product_offering']}\n"
                if startup_profile.get('target_market'):
                    report += f"**Target Market:** {startup_profile['target_market']}\n"
                if startup_profile.get('business_model'):
                    report += f"**Business Model:** {startup_profile['business_model']}\n"
                if startup_profile.get('go_to_market_strategy'):
                    report += f"**GTM Strategy:** {startup_profile['go_to_market_strategy']}\n"
                if startup_profile.get('total_funding'):
                    report += f"**Funding:** {startup_profile['total_funding']}\n"
                report += "\n"
            
            yield PipelineStatus("report_generation", "Generated product & strategy comparison", 0.8)
            
            report += f"""

## Market Positioning Comparison

"""
            
            for company_name in valid_companies:
                categories = all_market_categories[company_name]
                report += f"**{company_name}:** {', '.join(categories) if categories else 'General technology'}\n"
            
            report += f"""

## Key Differentiators

"""
            
            for company_name in valid_companies:
                startup_profile = all_startup_profiles[company_name]
                report += f"""
### {company_name}
"""
                differentiator = startup_profile.get('differentiator')
                competitive_advantage = startup_profile.get('competitive_advantage')
                
                if differentiator:
                    report += f"- {differentiator.strip()}\n"
                if competitive_advantage:
                    report += f"- {competitive_advantage.strip()}\n"
                if not differentiator and not competitive_advantage:
                    report += "- Differentiation not clearly articulated in available data\n"
            
            yield PipelineStatus("report_generation", "Generated differentiators section", 0.85)
            
            report += f"""

## Competitive Matrix

| Company | Market Focus | Product Offering | Key Strength | Funding Stage |
|---------|--------------|------------------|--------------|---------------|
"""
            
            for company_name in valid_companies:
                categories = all_market_categories[company_name]
                startup_profile = all_startup_profiles[company_name]
                
                market_focus = ', '.join(categories[:2]) if categories else 'Technology'
                product_offering = startup_profile.get('product_offering') or 'Not specified'
                if len(product_offering) > 40:
                    product_offering = product_offering[:40] + '...'
                
                key_strength = startup_profile.get('differentiator') or startup_profile.get('competitive_advantage') or 'Not specified'
                if len(key_strength) > 50:
                    key_strength = key_strength[:50] + '...'
                
                funding_stage = startup_profile.get('stage_round') or startup_profile.get('total_funding') or 'Unknown'
                
                report += f"| {company_name} | {market_focus} | {product_offering} | {key_strength} | {funding_stage} |\n"
            
            yield PipelineStatus("analysis", "Generating investment insights...", 0.9)
            
            report += f"""

## Investment Insights
"""
            
            # Calculate insights
            # Get all non-empty market categories
            category_sets = [set(all_market_categories[c]) for c in valid_companies if all_market_categories[c]]
            
            if len(category_sets) > 1:
                market_overlap = len(set.intersection(*category_sets)) > 0
            elif len(category_sets) == 1:
                market_overlap = False  # Only one company has categories
            else:
                market_overlap = False  # No companies have categories
                
            companies_with_differentiators = len([c for c in valid_companies if all_startup_profiles[c].get('differentiator') or all_startup_profiles[c].get('competitive_advantage')])
            companies_with_funding = len([c for c in valid_companies if all_startup_profiles[c].get('total_funding') or all_startup_profiles[c].get('stage_round')])
            companies_with_experienced_founders = len([c for c in valid_companies if any(f.get('past_companies') for f in all_founders[c])])
            
            report += f"""
- **Market Overlap:** {'High' if market_overlap else 'Low'} - Companies {'compete directly' if market_overlap else 'operate in different niches'}
- **Differentiation Clarity:** {companies_with_differentiators}/{len(valid_companies)} companies have clear differentiation
- **Funding Transparency:** {companies_with_funding}/{len(valid_companies)} companies have disclosed funding information
- **Founder Experience:** {companies_with_experienced_founders}/{len(valid_companies)} companies have founders with notable previous experience

## Strategic Recommendations
"""
            
            # Generate strategic insights
            high_growth_indicators = []
            risk_factors = []
            
            for company_name in valid_companies:
                startup_profile = all_startup_profiles[company_name]
                founders = all_founders[company_name]
                
                # Growth indicators
                if startup_profile.get('key_metrics'):
                    high_growth_indicators.append(f"{company_name} has disclosed metrics")
                if any(f.get('past_companies') for f in founders):
                    high_growth_indicators.append(f"{company_name} has experienced founders")
                if startup_profile.get('competitive_advantage'):
                    high_growth_indicators.append(f"{company_name} has clear competitive advantage")
                
                # Risk factors
                if not startup_profile.get('differentiator'):
                    risk_factors.append(f"{company_name} lacks clear differentiation")
                if not any(f.get('past_companies') for f in founders):
                    risk_factors.append(f"{company_name} founders lack notable experience")
            
            if high_growth_indicators:
                report += f"**Growth Indicators:**\n"
                for indicator in high_growth_indicators[:5]:
                    report += f"- {indicator}\n"
                report += "\n"
            
            if risk_factors:
                report += f"**Risk Factors:**\n"
                for risk in risk_factors[:5]:
                    report += f"- {risk}\n"
                report += "\n"
            
            report += f"""
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
            
            yield PipelineStatus("completion", f"Comparison report completed for {len(valid_companies)} companies", 1.0, data=report)
            
        except Exception as e:
            yield PipelineStatus("error", f"Error generating comparison report: {str(e)}", 0.0, error=str(e))

    # Keep the original non-streaming methods for backward compatibility
    async def generate_single_company_report(self, company_name):
        """Generate comprehensive report for a single company (non-streaming version)"""
        async for status in self.generate_single_company_report_stream(company_name):
            if status.stage == "completion":
                return status.data
            elif status.stage == "error":
                return f"Error: {status.error}"
        return "Report generation failed"
    
    async def generate_comparison_report(self, company_names_list):
        """Generate comparison report for multiple companies (non-streaming version)"""
        async for status in self.generate_comparison_report_stream(company_names_list):
            if status.stage == "completion":
                return status.data
            elif status.stage == "error":
                return f"Error: {status.error}"
        return "Report generation failed"

async def main():
    """Example usage of the streaming report generator"""
    generator = MarketReportGenerator()
    
    # Example 1: Single company report with streaming
    print("=== SINGLE COMPANY REPORT (STREAMING) ===")
    async for status in generator.generate_single_company_report_stream("100ms"):
        print(f"[{status.stage.upper()}] {status.message} ({status.progress:.1%})")
        if status.stage == "completion":
            print("\n--- FINAL REPORT ---")
            print(status.data[:500] + "..." if len(status.data) > 500 else status.data)
        elif status.stage == "error":
            print(f"ERROR: {status.error}")
    
    print("\n" + "="*80 + "\n")
    
    # Example 2: Comparison report with streaming
    print("=== COMPARISON REPORT (STREAMING) ===")
    comparison_companies = ["100ms", "11x.ai", "1stcollab"]
    async for status in generator.generate_comparison_report_stream(comparison_companies):
        print(f"[{status.stage.upper()}] {status.message} ({status.progress:.1%})")
        if status.stage == "completion":
            print("\n--- FINAL REPORT ---")
            print(status.data[:500] + "..." if len(status.data) > 500 else status.data)
        elif status.stage == "error":
            print(f"ERROR: {status.error}")

if __name__ == "__main__":
    asyncio.run(main())