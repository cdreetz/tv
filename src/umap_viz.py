import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import anthropic
import asyncio
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
import dash
from dash import dcc, html, Input, Output, callback
from src.utils import process_summaries_with_claude


class UMAPVisualizer:
    def __init__(self, csv_path, json_path, summaries_path=None):
        self.csv_path = csv_path
        self.json_path = json_path
        self.summaries_path = summaries_path
        self.df = None
        self.market_maps = None
        self.summaries = None
        self.client = anthropic.Anthropic()
        self.comparison_df = None
        
    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        with open(self.json_path, 'r') as f:
            self.market_maps = json.load(f)
        if self.summaries_path:
            with open(self.summaries_path, 'r') as f:
                self.summaries = json.load(f)
        
        # Add categories
        name_to_category = {}
        for category, companies in self.market_maps.items():
            for display_name, file_name in companies.items():
                norm_display = display_name.lower().replace(' ', '').replace('_', '').replace('.', '').replace('-', '')
                norm_file = file_name.lower().replace(' ', '').replace('_', '').replace('.', '').replace('-', '')
                name_to_category[norm_display] = category
                name_to_category[norm_file] = category
                
        self.df['category'] = self.df['name'].str.lower().str.replace(' ', '').str.replace('_', '').str.replace('.', '').str.replace('-', '').map(name_to_category).fillna('Other')
        
    def load_existing_claude_results(self, results_file="claude_results_filtered.json"):
        """Load existing Claude results from file"""
        try:
            with open(results_file, 'r') as f:
                claude_results = json.load(f)
            print(f"Loaded {len(claude_results)} existing Claude results from {results_file}")
            return claude_results
        except FileNotFoundError:
            # Try the original file if filtered doesn't exist
            try:
                with open("claude_results.json", 'r') as f:
                    claude_results = json.load(f)
                print(f"Loaded {len(claude_results)} existing Claude results from claude_results.json")
                return claude_results
            except FileNotFoundError:
                raise FileNotFoundError("No Claude results file found. Please run Claude processing first.")
    
    def filter_companies_with_market_labels(self, claude_results):
        """Filter to only companies that have market map labels"""
        # Get all company names from market maps
        market_map_companies = set()
        for category, companies in self.market_maps.items():
            for display_name, file_name in companies.items():
                # Normalize names for matching
                norm_display = display_name.lower().replace(' ', '').replace('_', '').replace('.', '').replace('-', '')
                norm_file = file_name.lower().replace(' ', '').replace('_', '').replace('.', '').replace('-', '')
                market_map_companies.add(norm_display)
                market_map_companies.add(norm_file)
        
        # Filter claude results to only include companies in market maps
        filtered_results = {}
        for company_name, data in claude_results.items():
            norm_name = company_name.lower().replace(' ', '').replace('_', '').replace('.', '').replace('-', '')
            if norm_name in market_map_companies:
                filtered_results[company_name] = data
        
        print(f"Filtered to {len(filtered_results)} companies with market map labels (from {len(claude_results)} total)")
        return filtered_results
        
    def extract_products_with_claude(self, batch_size=20, max_concurrent=5, limit=500, force_reprocess=False):
        """Extract product descriptions using Claude (limited to first N companies)"""
        # Take only the first N companies
        limited_summaries = dict(list(self.summaries.items())[:limit])
        print(f"Processing first {len(limited_summaries)} companies out of {len(self.summaries)} total")
        
        # If force_reprocess is True, use a different output file or delete existing results
        output_file = "claude_results_filtered.json" if force_reprocess else "claude_results.json"
        
        return asyncio.run(process_summaries_with_claude(limited_summaries, self.client, batch_size, max_concurrent, output_file))
        
    def create_new_umap_projections(self, claude_results, n_components=3, random_state=42):
        """Create new UMAP projections from extracted product descriptions"""
        # Filter to companies with known product descriptions (but keep all, not just market map ones)
        # Improved filtering to catch all variations of unknown descriptions
        known_products = {name: data for name, data in claude_results.items() 
                         if not (data['product_description'].lower() == 'unknown' or 
                                'unknown' in data['product_description'].lower())}
        
        if len(known_products) < 10:
            raise ValueError(f"Not enough companies with known products for UMAP (only {len(known_products)} found)")
        
        # Prepare text data
        companies = list(known_products.keys())
        descriptions = [known_products[name]['product_description'] for name in companies]
        
        print(f"Creating UMAP from {len(descriptions)} product descriptions...")
        
        # Create TF-IDF embeddings with better parameters
        vectorizer = TfidfVectorizer(
            max_features=min(1000, len(descriptions) * 10),  # Scale features with dataset size
            stop_words='english', 
            min_df=max(1, len(descriptions) // 100),  # Scale min_df with dataset size
            max_df=0.85, 
            ngram_range=(1, 2),  # Stick to unigrams and bigrams
            sublinear_tf=True,
            norm='l2'  # L2 normalization
        )
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # Create UMAP projection with parameters that create better scaled output
        n_neighbors = min(30, max(5, len(companies) // 10))  # Scale neighbors appropriately
        
        umap_model = UMAP(
            n_components=n_components, 
            random_state=random_state, 
            n_neighbors=n_neighbors,
            min_dist=0.1,  # Smaller min_dist for tighter clusters
            spread=1.0,
            metric='cosine',
            densmap=False,  # Disable densmap for more stable results
            output_metric='euclidean',  # Ensure euclidean output
            transform_seed=random_state  # Ensure reproducible transforms
        )
        
        # Fit and transform
        umap_embeddings = umap_model.fit_transform(tfidf_matrix.toarray())
        
        # Normalize the embeddings to a reasonable scale (-5 to +5)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        umap_embeddings_scaled = scaler.fit_transform(umap_embeddings)
        
        # Create new dataframe with projections
        new_df = pd.DataFrame({
            'name': companies,
            'z1_new': umap_embeddings_scaled[:, 0],
            'z2_new': umap_embeddings_scaled[:, 1],
            'z3_new': umap_embeddings_scaled[:, 2] if n_components == 3 else 0,
            'product_description': descriptions
        })
        
        # Add categories and market map status
        name_to_category = {}
        market_map_companies = set()
        
        for category, companies_dict in self.market_maps.items():
            for display_name, file_name in companies_dict.items():
                norm_display = display_name.lower().replace(' ', '').replace('_', '').replace('.', '').replace('-', '')
                norm_file = file_name.lower().replace(' ', '').replace('_', '').replace('.', '').replace('-', '')
                name_to_category[norm_display] = category
                name_to_category[norm_file] = category
                market_map_companies.add(norm_display)
                market_map_companies.add(norm_file)
                
        new_df['category'] = new_df['name'].str.lower().str.replace(' ', '').str.replace('_', '').str.replace('.', '').str.replace('-', '').map(name_to_category).fillna('No Market Map')
        
        # Add a column to track if company has market map label
        new_df['has_market_map'] = new_df['name'].str.lower().str.replace(' ', '').str.replace('_', '').str.replace('.', '').str.replace('-', '').isin(market_map_companies)
        
        market_map_count = new_df['has_market_map'].sum()
        print(f"Created projections for {len(new_df)} companies:")
        print(f"  - {market_map_count} with market map labels across {new_df[new_df['has_market_map']]['category'].nunique()} categories")
        print(f"  - {len(new_df) - market_map_count} without market map labels")
        print(f"Projection ranges: X: [{new_df['z1_new'].min():.2f}, {new_df['z1_new'].max():.2f}], Y: [{new_df['z2_new'].min():.2f}, {new_df['z2_new'].max():.2f}], Z: [{new_df['z3_new'].min():.2f}, {new_df['z3_new'].max():.2f}]")
        
        return new_df, umap_model, vectorizer
        
    def prepare_comparison_data(self, new_df):
        """Prepare merged data for interactive visualization"""
        # Merge original and new projections
        original_subset = self.df[self.df['name'].isin(new_df['name'])].copy()
        
        self.comparison_df = original_subset.merge(
            new_df[['name', 'z1_new', 'z2_new', 'z3_new', 'product_description']], 
            on='name', how='inner'
        )
        
        print(f"Prepared comparison data for {len(self.comparison_df)} companies")
        return self.comparison_df
        
    def launch_interactive_viz(self, port=8050):
        """Launch interactive Dash app for exploring projections"""
        if self.comparison_df is None:
            raise ValueError("No comparison data available. Run prepare_comparison_data() first.")
        
        app = dash.Dash(__name__)
        
        # Get unique categories for dropdown
        categories = sorted(self.comparison_df['category'].unique())
        
        app.layout = html.Div([
            html.H1("Interactive UMAP Visualization", style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.Label("Projection Type:"),
                    dcc.RadioItems(
                        id='projection-type',
                        options=[
                            {'label': 'Original (Summary-based)', 'value': 'original'},
                            {'label': 'New (Product-based)', 'value': 'new'}
                        ],
                        value='original',
                        inline=True
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Show Companies:"),
                    dcc.RadioItems(
                        id='market-map-filter',
                        options=[
                            {'label': 'All Companies', 'value': 'all'},
                            {'label': 'Only Market Map Companies', 'value': 'market_only'},
                            {'label': 'Only Non-Market Map Companies', 'value': 'non_market_only'}
                        ],
                        value='all',
                        inline=True
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    html.Label("Categories:"),
                    dcc.Dropdown(
                        id='category-filter',
                        options=[{'label': 'All Categories', 'value': 'all'}] + 
                               [{'label': cat, 'value': cat} for cat in categories],
                        value='all',
                        multi=True
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Point Size:"),
                    dcc.Slider(
                        id='point-size',
                        min=1,
                        max=10,
                        step=1,
                        value=4,
                        marks={i: str(i) for i in range(1, 11)}
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ], style={'margin': '20px 0'}),
            
            dcc.Graph(id='umap-plot', style={'height': '80vh'}),
            
            html.Div(id='stats-display', style={'margin': '20px', 'textAlign': 'center'})
        ])
        
        @app.callback(
            [Output('umap-plot', 'figure'),
             Output('stats-display', 'children')],
            [Input('projection-type', 'value'),
             Input('market-map-filter', 'value'),
             Input('category-filter', 'value'),
             Input('point-size', 'value')]
        )
        def update_plot(projection_type, market_map_filter, selected_categories, point_size):
            # Filter data based on market map selection
            filtered_df = self.comparison_df.copy()
            
            if market_map_filter == 'market_only':
                filtered_df = filtered_df[filtered_df['has_market_map'] == True]
            elif market_map_filter == 'non_market_only':
                filtered_df = filtered_df[filtered_df['has_market_map'] == False]
            
            # Filter data based on category selection
            if selected_categories != 'all' and selected_categories:
                if isinstance(selected_categories, str):
                    selected_categories = [selected_categories]
                filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
            
            # Choose coordinates based on projection type
            if projection_type == 'original':
                x_col, y_col, z_col = 'z1', 'z2', 'z3'
                title = f"Original UMAP (Summary-based) - {len(filtered_df)} companies"
            else:
                x_col, y_col, z_col = 'z1_new', 'z2_new', 'z3_new'
                title = f"New UMAP (Product-based) - {len(filtered_df)} companies"
            
            # Create 3D scatter plot
            fig = px.scatter_3d(
                filtered_df, 
                x=x_col, y=y_col, z=z_col,
                color='category',
                hover_data=['name', 'product_description'] if 'product_description' in filtered_df.columns else ['name'],
                title=title,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_traces(marker=dict(size=point_size))
            fig.update_layout(
                height=700,
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            # Stats display
            market_count = filtered_df['has_market_map'].sum() if 'has_market_map' in filtered_df.columns else 0
            non_market_count = len(filtered_df) - market_count
            stats = f"Showing {len(filtered_df)} companies ({market_count} with market maps, {non_market_count} without) across {filtered_df['category'].nunique()} categories"
            
            return fig, stats
        
        print(f"Starting interactive visualization at http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        app.run(debug=True, port=port, host='127.0.0.1')
        
    def save_new_projections(self, new_df, filename="new_umap_projections.csv"):
        """Save new projections to CSV"""
        new_df.to_csv(filename, index=False)
        print(f"New projections saved to {filename}")
        
    def filter_known_products(self, claude_results):
        """Filter out companies with unknown product descriptions"""
        product_map = {name.lower().replace(' ', '').replace('_', '').replace('.', '').replace('-', ''): data['product_description'] 
                      for name, data in claude_results.items()}
        
        self.df['product_description'] = self.df['name'].str.lower().str.replace(' ', '').str.replace('_', '').str.replace('.', '').str.replace('-', '').map(product_map)
        # Improved filtering to catch all variations of unknown descriptions (case-insensitive)
        return self.df[~self.df['product_description'].str.lower().str.contains('unknown', na=False)]
        
    def plot(self, filtered_df=None, width=1400, height=1000, point_size=2, title="3D UMAP Projection"):
        plot_df = filtered_df if filtered_df is not None else self.df
        fig = px.scatter_3d(plot_df, x='z1', y='z2', z='z3', color='category', hover_data=['name'], title=title)
        fig.update_layout(width=width, height=height, showlegend=True)
        fig.update_traces(marker=dict(size=point_size))
        fig.show()

if __name__ == "__main__":
    viz = UMAPVisualizer(
        csv_path='8d7b3ce6f4596ddf83d6d955017a8210/3d_projection_emb.csv',
        json_path='8d7b3ce6f4596ddf83d6d955017a8210/market_maps_examples.json',
        summaries_path='8d7b3ce6f4596ddf83d6d955017a8210/company_llm_summaries.json'
    )
    viz.load_data()
    
    # Load existing Claude results instead of reprocessing
    claude_results = viz.load_existing_claude_results()
    
    # Create new UMAP projections from product descriptions
    new_df, umap_model, vectorizer = viz.create_new_umap_projections(claude_results)
    
    # Prepare comparison data and launch interactive visualization
    viz.prepare_comparison_data(new_df)
    viz.launch_interactive_viz()
    
    # Save new projections
    viz.save_new_projections(new_df)
