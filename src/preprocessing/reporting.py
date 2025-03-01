import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging

# Configure logger
logger = logging.getLogger(__name__)

class FeatureReporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def _format_axis_labels(self, ax, x_rotation=90):
        """Helper method to format axis labels consistently"""
        ax.tick_params(axis='x', rotation=x_rotation)
        ax.tick_params(axis='y', labelsize=10)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.title.set_size(14)
        
        # Format large numbers with K/M suffix
        if ax.get_ylabel() in ['TOTAL_VALUE', 'price_per_sqft']:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}' if x < 1000 
                                                         else f'${x/1000:,.0f}K' if x < 1000000 
                                                         else f'${x/1000000:,.1f}M'))
    
    def generate_combined_plots(self, df):
        # Calculate total value
        df['TOTAL_VALUE'] = df['BLDG_VALUE'].fillna(0) + df['LAND_VALUE'].fillna(0)
        
        # Set modern style with enhanced aesthetics
        plt.style.use('seaborn-v0_8-white')
        sns.set_palette("husl")
        
        # Set up the figure with improved spacing and modern style
        plt.rcParams.update({
            'figure.constrained_layout.use': True,
            'figure.constrained_layout.h_pad': 1.0,
            'figure.constrained_layout.w_pad': 1.0,
            'font.family': 'sans-serif',
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'axes.grid': True,
            'grid.alpha': 0.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.titleweight': 'bold',
            'figure.facecolor': 'white',
            'axes.facecolor': '#f8f9fa',
            'axes.edgecolor': '#343a40',
            'axes.labelcolor': '#343a40',
            'xtick.color': '#343a40',
            'ytick.color': '#343a40',
            'grid.color': '#dee2e6'
        })
        
        # Create figure with 2x2 subplots and better spacing
        fig = plt.figure(figsize=(40, 30), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, hspace=1.0, wspace=0.6)
        
        # Increase font sizes for better readability
        plt.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 24,
            'axes.labelsize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16
        })
        # 1. Total Value Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data=df, x='TOTAL_VALUE', bins=50, ax=ax1, color='#2ecc71', 
                    edgecolor='white', alpha=0.8)
        ax1.set_facecolor('#f8f9fa')
        ax1.set_xscale('log')
        ax1.set_title('Distribution of Property Values', pad=20, fontweight='bold')
        ax1.set_xlabel('Property Value (Log Scale)')
        ax1.set_ylabel('Number of Properties')
        
        # Helper function to format currency values
        def format_currency(x):
            if pd.isna(x):
                return 'N/A'
            if x >= 1e6:
                return f'${x/1e6:.1f}M'
            elif x >= 1e3:
                return f'${x/1e3:.0f}K'
            else:
                return f'${x:.0f}'
        
        stats_text = f"""
        Statistics:
        Total Value (Median): {format_currency(df['TOTAL_VALUE'].median())}
        Building Value (Median): {format_currency(df['BLDG_VALUE'].median())}
        Land Value (Median): {format_currency(df['LAND_VALUE'].median())}
        """
        ax1.text(0.95, 0.95, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='#343a40', boxstyle='round,pad=0.5'),
                fontsize=11)
        
        # Add median and mean lines with enhanced styling
        median_val = df['TOTAL_VALUE'].median()
        mean_val = df['TOTAL_VALUE'].mean()
        ax1.axvline(x=median_val, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=2, label='Median')
        ax1.axvline(x=mean_val, color='#3498db', linestyle='--', alpha=0.8, linewidth=2, label='Mean')
        ax1.legend()
        self._format_axis_labels(ax1)
        
        # 2. Living Area vs Building Value 
        ax2 = fig.add_subplot(gs[0, 1])
        sns.scatterplot(data=df, x='LIVING_AREA', y='BLDG_VALUE', 
                       alpha=0.5, ax=ax2)
        ax2.set_title('Living Area vs Building Value', pad=20, fontweight='bold')
        ax2.set_xlabel('Living Area (sq ft)')
        ax2.set_ylabel('Building Value ($)')
        
        # Add trend line with formatted equation and error handling
        try:
            # Filter out NaN and infinite values
            mask = np.isfinite(df['LIVING_AREA']) & np.isfinite(df['BLDG_VALUE'])
            if mask.sum() > 1:  # Need at least 2 points for linear fit
                z = np.polyfit(df.loc[mask, 'LIVING_AREA'], 
                             df.loc[mask, 'BLDG_VALUE'], 1)
                p = np.poly1d(z)
                price_per_sqft = z[0]
                if price_per_sqft >= 1000:
                    trend_label = f'Trend: ${price_per_sqft/1000:.1f}K/sq.ft'
                else:
                    trend_label = f'Trend: ${price_per_sqft:.0f}/sq.ft'
                ax2.plot(df.loc[mask, 'LIVING_AREA'], 
                        p(df.loc[mask, 'LIVING_AREA']), 
                        linestyle='--', color='red', alpha=0.8,
                        label=trend_label)
        except Exception as e:
            logger.warning(f"Could not generate trend line: {str(e)}")
        ax2.legend()
        
        # 3. Room Analysis
        ax3 = fig.add_subplot(gs[1, 0])
        # Convert to integer before counting
        df['TT_RMS'] = pd.to_numeric(df['TT_RMS'], errors='coerce').round().astype('Int64')
        df['BED_RMS'] = pd.to_numeric(df['BED_RMS'], errors='coerce').round().astype('Int64')
        room_data = pd.DataFrame({
            'Total Rooms': df['TT_RMS'].value_counts().sort_index(),
            'Bedrooms': df['BED_RMS'].value_counts().sort_index()
        })
        room_data.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Room Distribution', pad=20, fontweight='bold')
        ax3.set_xlabel('Number of Rooms')
        ax3.set_ylabel('Number of Properties')
        ax3.legend(title='Room Type')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # 4. Correlation Matrix
        ax4 = fig.add_subplot(gs[1, 1])
        numeric_cols = ['BLDG_VALUE', 'LAND_VALUE', 'LIVING_AREA', 'GROSS_AREA',
                       'TT_RMS', 'BED_RMS', 'FULL_BTH', 'HLF_BTH', 'LAND_SF']
        # Filter to only include columns that exist in the dataframe
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) > 1:  # Need at least 2 columns for correlation
            # Ensure numeric columns are float type
            df_numeric = df[available_cols].apply(pd.to_numeric, errors='coerce')
            # Calculate correlations
            correlation = df_numeric.corr()
            # Sort by correlation with BLDG_VALUE if it exists, otherwise take first column
            sort_col = 'BLDG_VALUE' if 'BLDG_VALUE' in correlation.columns else correlation.columns[0]
            sorted_cols = correlation[sort_col].abs().sort_values(ascending=False).index
            correlation = correlation.loc[sorted_cols, sorted_cols]

            # Create readable labels
            label_map = {
                'BLDG_VALUE': 'Building Value',
                'LAND_VALUE': 'Land Value',
                'LIVING_AREA': 'Living Area',
                'GROSS_AREA': 'Gross Area',
                'TT_RMS': 'Total Rooms',
                'BED_RMS': 'Bedrooms',
                'FULL_BTH': 'Full Baths',
                'LAND_SF': 'Land Sq.Ft',
                'HLF_BTH': 'Half Baths'
            }
            
            # Create labels using only available columns
            labels = [label_map.get(col, col) for col in correlation.columns]
            
            # Create correlation matrix with better labels
            try:
                sns.heatmap(correlation, 
                           mask=np.triu(np.ones_like(correlation, dtype=bool)),
                           annot=True,
                           fmt='.2f',
                           cmap='RdYlBu_r',
                           center=0,
                           ax=ax4,
                           annot_kws={'size': 10},
                           square=True,
                           xticklabels=labels,
                           yticklabels=labels,
                           cbar_kws={'label': 'Correlation Coefficient'})
            except Exception as e:
                logger.warning(f"Could not generate heatmap: {str(e)}")
                ax4.text(0.5, 0.5, 'Could not generate correlation matrix',
                        ha='center', va='center')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for correlation matrix',
                    ha='center', va='center')
        
        # Rotate labels for better fit
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax4.get_yticklabels(), rotation=0)
        ax4.set_title('Feature Correlation Matrix', pad=20, fontweight='bold')
            
        
        # Save the combined figure
        plt.savefig(f'{self.output_dir}/feature_analysis.png', dpi=300, facecolor='white')
        plt.close()
    
    def generate_feature_summary(self, df):
        # Get feature categories
        engineered_features = [
            'price_per_sqft', 'land_value_ratio', 'building_value_ratio',
            'property_age', 'age_category', 'total_bathrooms',
            'non_living_area', 'living_area_ratio', 'has_renovation',
            'years_since_renovation'
        ]
        
        # Calculate basic statistics
        summary = pd.DataFrame({
            'Feature': df.columns,
            'Type': df.dtypes,
            'Missing_Values': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
            'Unique_Values': df.nunique(),
            'Mean': df.mean(numeric_only=True),
            'Std': df.std(numeric_only=True),
            'Min': df.min(numeric_only=True),
            'Max': df.max(numeric_only=True)
        })
        
        # Add feature category
        summary['Category'] = 'Original'
        summary.loc[summary['Feature'].isin(engineered_features), 'Category'] = 'Engineered'
        
        # Save detailed summary
        summary.to_csv(f'{self.output_dir}/feature_summary.csv', index=False)
        
        # Create enhanced markdown report
        with open(f'{self.output_dir}/feature_report.md', 'w') as f:
            f.write('# Feature Analysis Report\n\n')
            
            # Dataset Overview
            f.write('## Dataset Overview\n')
            f.write('### Data Structure\n')
            f.write(f'- Total Variables: {len(df.columns)}\n')
            f.write(f'- Total Samples: {len(df)}\n\n')
            
            f.write('### Variable Types\n')
            f.write(f'- Input Features: {len(df.columns) - 1}\n')  # Assuming one target column
            f.write(f'  - Raw Features: {len(summary[summary.Category == "Original"])}\n')
            f.write(f'  - Engineered Features: {len(summary[summary.Category == "Engineered"])}\n')
            f.write(f'- Target Variable: 1\n\n')
            
            f.write('### Data Types\n')
            f.write(f'- Numerical Variables: {len(df.select_dtypes(include=["int64", "float64"]).columns)}\n')
            f.write(f'- Categorical Variables: {len(df.select_dtypes(include=["object", "category"]).columns)}\n\n')
            
            # Missing Values Analysis
            f.write('## Missing Values Analysis\n')
            missing = summary[summary['Missing_Values'] > 0].sort_values('Missing_Percentage', ascending=False)
            if not missing.empty:
                f.write('### Features with Missing Values (Sorted by Percentage)\n')
                f.write(missing[['Feature', 'Category', 'Missing_Values', 'Missing_Percentage']].to_markdown() + '\n\n')
            else:
                f.write('No missing values found in the dataset.\n\n')
            
            # Numerical Features Analysis
            f.write('## Numerical Features Analysis\n')
            numerical = summary[summary['Type'].isin(['int64', 'float64'])].sort_values('Category')
            if not numerical.empty:
                for category in ['Original', 'Engineered']:
                    cat_numerical = numerical[numerical['Category'] == category]
                    if not cat_numerical.empty:
                        f.write(f'### {category} Numerical Features\n')
                        stats_table = cat_numerical[['Feature', 'Mean', 'Std', 'Min', 'Max', 'Missing_Percentage']]
                        stats_table = stats_table.round(2)
                        f.write(stats_table.to_markdown() + '\n\n')
            
            # Categorical Features Analysis
            categorical = summary[summary['Type'] == 'object'].sort_values('Category')
            if not categorical.empty:
                f.write('## Categorical Features Analysis\n')
                for category in ['Original', 'Engineered']:
                    cat_categorical = categorical[categorical['Category'] == category]
                    if not cat_categorical.empty:
                        f.write(f'### {category} Categorical Features\n')
                        f.write(cat_categorical[['Feature', 'Unique_Values', 'Missing_Percentage']].to_markdown() + '\n\n')
    
    def generate_all_reports(self, df):
        self.generate_combined_plots(df)
        self.generate_feature_summary(df)
