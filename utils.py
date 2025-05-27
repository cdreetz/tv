import json
import asyncio
import time
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


async def process_summaries_with_claude(summaries, client, batch_size=20, max_concurrent=5, 
                                      output_file="claude_results.json", max_retries=3):
    """Async utility to process company summaries with Claude, with persistence and rate limiting"""
    
    # Load existing results if file exists
    results_file = Path(output_file)
    if results_file.exists():
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        print(f"Loaded {len(existing_results)} existing results from {output_file}")
    else:
        existing_results = {}
    
    # Filter out already processed companies
    remaining_summaries = {k: v for k, v in summaries.items() if k not in existing_results}
    print(f"Processing {len(remaining_summaries)} remaining companies (out of {len(summaries)} total)")
    
    if not remaining_summaries:
        print("All companies already processed!")
        return existing_results
    
    async def process_batch_with_retry(batch, batch_num, total_batches):
        """Process a batch with retry logic and rate limiting"""
        
        for attempt in range(max_retries):
            try:
                # Rate limiting - wait between attempts
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retrying batch {batch_num} (attempt {attempt + 1}) after {wait_time}s...")
                    await asyncio.sleep(wait_time)
                
                prompt = "Extract product/service descriptions from these company summaries. For each company, if no clear product is described, use 'unknown'.\n\n"
                
                for j, (company_name, summary) in enumerate(batch):
                    prompt += f"{j+1}. Company: {company_name}\nSummary: {summary}\n\n"
                
                prompt += f"""Respond with only a JSON array with {len(batch)} objects in this exact format:
[
  {{"company_name": "Company 1", "product_description": "description here"}},
  {{"company_name": "Company 2", "product_description": "description here"}}
]"""
                
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    response = await loop.run_in_executor(
                        executor,
                        lambda: client.messages.create(
                            model="claude-3-5-haiku-latest",
                            max_tokens=2000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                    )
                
                response_text = response.content[0].text.strip()
                if response_text.startswith('```'):
                    response_text = response_text.split('\n', 1)[1].rsplit('\n', 1)[0]
                
                batch_results = json.loads(response_text)
                results = {}
                for result in batch_results:
                    results[result['company_name']] = {"product_description": result['product_description']}
                
                print(f"✓ Completed batch {batch_num}/{total_batches}")
                return results
                
            except Exception as e:
                print(f"✗ Batch {batch_num} attempt {attempt + 1} failed: {e}")
                
                # Check if it's a rate limit error
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait_time = 60  # Wait longer for rate limits
                    print(f"Rate limit detected, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                
                if attempt == max_retries - 1:
                    # Final attempt failed, mark all as unknown
                    print(f"✗ Batch {batch_num} failed after {max_retries} attempts")
                    return {company_name: {"product_description": "unknown"} for company_name, _ in batch}
        
        return {}
    
    def save_results(current_results):
        """Save current results to file"""
        with open(results_file, 'w') as f:
            json.dump(current_results, f, indent=2)
    
    companies = list(remaining_summaries.items())
    batches = [companies[i:i + batch_size] for i in range(0, len(companies), batch_size)]
    
    # Process batches with limited concurrency and save progress
    semaphore = asyncio.Semaphore(max_concurrent)
    all_results = existing_results.copy()
    
    async def process_with_semaphore_and_save(batch, batch_num):
        async with semaphore:
            # Add small delay between batches to be nice to the API
            await asyncio.sleep(0.5)
            
            batch_result = await process_batch_with_retry(batch, batch_num + 1, len(batches))
            
            # Save progress after each batch
            all_results.update(batch_result)
            save_results(all_results)
            
            return batch_result
    
    # Process all batches
    tasks = [process_with_semaphore_and_save(batch, i) for i, batch in enumerate(batches)]
    await asyncio.gather(*tasks)
    
    print(f"✓ Processing complete! Results saved to {output_file}")
    print(f"Total companies processed: {len(all_results)}")
    
    return all_results