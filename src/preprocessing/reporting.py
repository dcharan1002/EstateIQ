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
        df = df.copy()
        
        try:
            # Handle empty dataframe
            if df.empty:
                df['TOTAL_VALUE'] = pd.Series(dtype='float64')
            else:
                # Safely calculate total value if components exist
                df['TOTAL_VALUE'] = 0  # Initialize with zeros
                if 'BLDG_VALUE' in df.columns:
                    df['TOTAL_VALUE'] += pd.to_numeric(df['BLDG_VALUE'], errors='coerce').fillna(0)
                if 'LAND_VALUE' in df.columns:
                    df['TOTAL_VALUE'] += pd.to_numeric(df['LAND_VALUE'], errors='coerce').fillna(0)
                
                # If no value columns exist, create empty total value
                if 'BLDG_VALUE' not in df.columns and 'LAND_VALUE' not in df.columns:
                    df['TOTAL_VALUE'] = pd.Series(dtype='float64')
            
            # Set style and create plots
            plt.style.use('seaborn-v0_8-white')
            sns.set_palette("husl")
            
            # Create and save the plots
            fig = plt.figure(figsize=(40, 30))
            plt.subplots_adjust(hspace=0.4, wspace=0.4)
            
            # Create all subplots
            for i in range(4):
                plt.subplot(2, 2, i+1)
                if i == 0:  # Value distribution
                    self._create_value_distribution_plot(df)
                elif i == 1:  # Area vs Value
                    self._create_area_value_plot(df)
                elif i == 2:  # Room distribution
                    self._create_room_distribution_plot(df)
                else:  # Correlation matrix
                    self._create_correlation_matrix(df)
            
            # Save the figure
            plt.savefig(f'{self.output_dir}/feature_analysis.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            # Create a simple error plot
            plt.figure(figsize=(10, 5))
            plt.text(0.5, 0.5, f'Error generating plots: {str(e)}', 
                    ha='center', va='center')
            plt.savefig(f'{self.output_dir}/feature_analysis.png')
            plt.close()
    
    def _create_value_distribution_plot(self, df):
        """Create value distribution subplot"""
        if not df['TOTAL_VALUE'].empty and df['TOTAL_VALUE'].notna().any():
            sns.histplot(data=df, x='TOTAL_VALUE', bins=50, 
                        color='#2ecc71', edgecolor='white')
            plt.xscale('log')
        else:
            plt.text(0.5, 0.5, 'No value data available',
                    ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Property Value Distribution')
        plt.xlabel('Total Value (log scale)')
        plt.ylabel('Count')
    
    def _create_area_value_plot(self, df):
        """Create area vs value subplot"""
        if all(col in df.columns for col in ['LIVING_AREA', 'BLDG_VALUE']):
            valid_mask = df['LIVING_AREA'].notna() & df['BLDG_VALUE'].notna()
            if valid_mask.any():
                sns.scatterplot(data=df[valid_mask], 
                              x='LIVING_AREA', y='BLDG_VALUE')
            else:
                plt.text(0.5, 0.5, 'No valid data points',
                        ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, 'Missing required columns',
                    ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Living Area vs Building Value')
    
    def _create_room_distribution_plot(self, df):
        """Create room distribution subplot"""
        room_cols = ['TT_RMS', 'BED_RMS']
        room_data = {}
        
        for col in room_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce')
                if series.notna().any():
                    room_data[col] = series.value_counts().sort_index()
        
        if room_data:
            pd.DataFrame(room_data).plot(kind='bar')
            plt.title('Room Distribution')
        else:
            plt.text(0.5, 0.5, 'No room data available',
                    ha='center', va='center', transform=plt.gca().transAxes)
    
    def _create_correlation_matrix(self, df):
        """Create correlation matrix subplot"""
        numeric_cols = ['BLDG_VALUE', 'LAND_VALUE', 'LIVING_AREA', 'GROSS_AREA',
                       'TT_RMS', 'BED_RMS', 'FULL_BTH', 'HLF_BTH']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) > 1:
            corr_df = df[available_cols].apply(pd.to_numeric, errors='coerce').corr()
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlations')
        else:
            plt.text(0.5, 0.5, 'Insufficient data for correlation',
                    ha='center', va='center', transform=plt.gca().transAxes)
    
    def generate_feature_summary(self, df):
        """Generate feature summary statistics and reports"""
        try:
            if df.empty:
                summary = pd.DataFrame(columns=['Feature', 'Type', 'Missing_Values', 
                                             'Missing_Percentage', 'Unique_Values', 
                                             'Mean', 'Std', 'Min', 'Max', 'Category'])
            else:
                # Get feature categories
                engineered_features = [
                    'price_per_sqft', 'land_value_ratio', 'building_value_ratio',
                    'property_age', 'age_category', 'total_bathrooms',
                    'non_living_area', 'living_area_ratio', 'has_renovation',
                    'years_since_renovation'
                ]
                
                # Calculate statistics
                stats = {col: self._calculate_column_stats(df[col]) for col in df.columns}
                
                # Create summary DataFrame
                summary = pd.DataFrame({
                    'Feature': df.columns,
                    'Type': df.dtypes,
                    'Missing_Values': df.isnull().sum(),
                    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
                    'Unique_Values': df.nunique(),
                    'Mean': [stats[col]['mean'] for col in df.columns],
                    'Std': [stats[col]['std'] for col in df.columns],
                    'Min': [stats[col]['min'] for col in df.columns],
                    'Max': [stats[col]['max'] for col in df.columns]
                })
                
                # Add categories
                summary['Category'] = 'Original'
                summary.loc[summary['Feature'].isin(engineered_features), 'Category'] = 'Engineered'
            
            # Save summary to CSV
            summary.to_csv(f'{self.output_dir}/feature_summary.csv', index=False)
            
            # Generate markdown report
            self._generate_markdown_report(df, summary)
            
        except Exception as e:
            logger.error(f"Error generating feature summary: {str(e)}")
            # Create minimal summary
            summary = pd.DataFrame({'Feature': ['Error'], 'Type': ['N/A']})
            summary.to_csv(f'{self.output_dir}/feature_summary.csv', index=False)
            
            with open(f'{self.output_dir}/feature_report.md', 'w') as f:
                f.write(f'# Error in Feature Analysis\n\nError: {str(e)}')
    
    def _calculate_column_stats(self, series):
        """Calculate statistics for a column with proper error handling"""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            return {
                'mean': numeric_series.mean(),
                'std': numeric_series.std(),
                'min': numeric_series.min(),
                'max': numeric_series.max()
            }
        except:
            return {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan
            }
    
    def _generate_markdown_report(self, df, summary):
        """Generate markdown report with proper formatting"""
        with open(f'{self.output_dir}/feature_report.md', 'w') as f:
            f.write('# Feature Analysis Report\n\n')
            
            # Dataset Overview
            f.write('## Dataset Overview\n')
            f.write(f'- Total Features: {len(df.columns)}\n')
            f.write(f'- Total Samples: {len(df)}\n')
            f.write(f'- Original Features: {len(summary[summary.Category == "Original"])}\n')
            f.write(f'- Engineered Features: {len(summary[summary.Category == "Engineered"])}\n\n')
            
            # Missing Values
            missing = summary[summary['Missing_Values'] > 0]
            if not missing.empty:
                f.write('## Missing Values\n')
                missing_table = missing[['Feature', 'Missing_Values', 'Missing_Percentage']]
                missing_table['Missing_Percentage'] = missing_table['Missing_Percentage'].map('{:.1f}%'.format)
                f.write(missing_table.to_markdown(index=False) + '\n\n')
            
            # Numeric Features
            numeric_cols = summary[summary['Type'].isin(['int64', 'float64'])]
            if not numeric_cols.empty:
                f.write('## Numeric Features\n')
                numeric_table = numeric_cols[['Feature', 'Mean', 'Std', 'Min', 'Max']]
                for col in ['Mean', 'Std', 'Min', 'Max']:
                    numeric_table[col] = numeric_table[col].apply(lambda x: 
                        f'{x:,.2f}' if pd.notnull(x) else 'N/A')
                f.write(numeric_table.to_markdown(index=False) + '\n\n')
    
    def generate_all_reports(self, df):
        """Generate all reports with error handling"""
        try:
            self.generate_combined_plots(df)
            self.generate_feature_summary(df)
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
            # Create error report
            with open(f'{self.output_dir}/error_report.md', 'w') as f:
                f.write(f'# Error Generating Reports\n\nError: {str(e)}')
