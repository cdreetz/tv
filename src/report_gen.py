import json
import asyncio
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.big_scraper import MarketMap
from dotenv import load_dotenv
from typing import Dict, List

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
    
    def find_similar_companies(self, company_name, top_k=5):
        """Find most similar companies to the given company"""
        if company_name not in self.company_names:
            return []
        
        company_idx = self.company_names.index(company_name)
        similarities = self.similarity_matrix[company_idx]
        
        # Get indices of most similar companies (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        similar_companies = []
        for idx in similar_indices:
            similar_companies.append({
                'name': self.company_names[idx],
                'similarity_score': similarities[idx],
                'summary': self.companies[self.company_names[idx]]['company_summary']
            })
        
        return similar_companies
    
    async def generate_single_company_report(self, company_name):
        """Generate comprehensive report for a single company"""
        if company_name not in self.companies:
            return f"Company '{company_name}' not found in database."
        
        # Enrich the company data with structured extraction
        print(f"Enriching data for {company_name}...")
        enriched_data = await self.market_map.enrich_company_data(company_name)
        
        # Extract structured data
        structured_data = enriched_data.get('structured_data', {})
        founders_info = structured_data.get('founders', [])
        startup_profile = structured_data.get('startup_profile', {})
        market_categories = structured_data.get('market_category', [])
        
        # Find similar companies
        similar_companies = self.find_similar_companies(company_name, top_k=4)
        
        # Get social links if available
        social_links = enriched_data.get('social_links', {})
        
        # Generate report
        report = f"""
        # Market Analysis Report: {company_name}

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

            ## Competitive Landscape
            The following companies operate in similar spaces:

        """
        
        for i, similar in enumerate(similar_companies, 1):
            report += f"""
        ### {i}. {similar['name']} 
        **Similarity Score:** {similar['similarity_score']:.3f}
        **Brief:** {similar['summary'][:200]}...

        """
        
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
        
        return report
    
    async def generate_comparison_report(self, company_names_list):
        """Generate comparison report for multiple companies"""
        valid_companies = [name for name in company_names_list if name in self.companies]
        
        if not valid_companies:
            return "No valid companies found in the provided list."
        
        # Enrich all companies with structured data
        enriched_companies = {}
        for company_name in valid_companies:
            print(f"Enriching data for {company_name}...")
            enriched_companies[company_name] = await self.market_map.enrich_company_data(company_name)
        
        # Extract structured data for all companies
        all_founders = {}
        all_startup_profiles = {}
        all_market_categories = {}
        
        for company_name in valid_companies:
            enriched_data = enriched_companies[company_name]
            structured_data = enriched_data.get('structured_data', {})
            
            all_founders[company_name] = structured_data.get('founders', [])
            all_startup_profiles[company_name] = structured_data.get('startup_profile', {})
            all_market_categories[company_name] = structured_data.get('market_category', [])
        
        # Generate comparison report
        report = f"""
        # Competitive Analysis Report

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
        
        return report

async def main():
    """Example usage of the report generator"""
    generator = MarketReportGenerator()
    
    # Example 1: Single company report
    #print("=== SINGLE COMPANY REPORT ===")
    #single_report = await generator.generate_single_company_report("100ms")
    #print(single_report)
    
    #print("\n" + "="*80 + "\n")
    
    # Example 2: Comparison report
    print("=== COMPARISON REPORT ===")
    comparison_companies = ["100ms", "11x.ai", "1stcollab"]
    comparison_report = await generator.generate_comparison_report(comparison_companies)
    print(comparison_report)

if __name__ == "__main__":
    asyncio.run(main())