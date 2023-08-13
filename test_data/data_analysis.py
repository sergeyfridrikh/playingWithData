import pandas as pd
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import ceil


def plot_ab_resistance(data):
    """Creates a plot of percentages of samples resistant/susceptible to antibiotics.

        Calculates percentages of samples from the data frame R/S to antibiotics .
        Uses matplotlib barh function to plot the results.

        Args:
            A dataframe with data on AB resistance.
            Each row is a sample, each column is an antibiotic.

        Returns:
            A matplotlib horizontal bar chart for AB resistance percentages.
    """

    # Count numbers of samples resistant (R) and susceptible (S) to each antibiotic
    rs_counts = data.apply(pd.Series.value_counts)

    # Calculate percentages of samples resistant (R) and susceptible (S) to each antibiotic
    rs_counts_r = rs_counts.loc['R'] / (rs_counts.loc['R'] + rs_counts.loc['S']) * 100
    rs_counts_s = rs_counts.loc['S'] / (rs_counts.loc['R'] + rs_counts.loc['S']) * 100

    values1 = np.array(rs_counts_r)
    values2 = np.array(rs_counts_s)

    # Plot horizontal bar plot demonstrating resistance/susceptibility of samples
    # to each antibiotic on the list
    fig, ax = plt.subplots(figsize=(10, 8))

    ind = np.arange(len(values1))
    width = 0.8

    plt.barh(ind, values1, width, color='darkorange')
    plt.barh(ind, values2, width, left=values1, color='green')

    # Define the tick format
    extra_space = 0.5
    ax.set_yticks(ind + width - extra_space)
    ax.set_yticklabels(rs_counts_r.index.values)
    ax.yaxis.set_tick_params(length=0, labelbottom=True)

    # Add resistant/susceptible percentage values to the plot
    for i, (v1, v2) in enumerate(zip(values1, values2)):
        plt.text(v1 * 0.45, i + .0, str(round(v1)) + "%", color='white',
                 fontweight='bold', fontsize=12, ha='center', va='center')
        plt.text(v1 + v2 * 0.45, i + .0, str(round(v2)) + "%", color='white',
                 fontweight='bold', fontsize=12, ha='center', va='center')

    plt.margins(x=0, y=0.02)

    # Add plot title and axis labels
    plt.title("Percentages of Samples Resistant and Susceptible \n" +
              "to the Antibiotics Used in the Study", fontsize=18, style="italic")
    plt.xlabel('Percentages of Antibiotic Resistant (orange) and Susceptible (green) Samples')
    plt.ylabel('Antibiotics')

    return plt

def plot_gene_counts(data):
    """Creates a plot of numbers of samples containing AR genes.

        Calculates numbers of samples containing AR genes.
        Uses matplotlib bar function to plot the results.

        Args:
            A dataframe with data on AR genes detected in the samples.
            Each row is a sample, each column is an AR gene.

        Returns:
            A matplotlib bar chart for numbers of samples containing AR genes.
    """
    # Count numbers of gene detections
    data_sum = data.sum(axis=0)

    values = np.array(data_sum)

    # Plot a bar plot demonstrating presence of antibiotic resistance genes
    # in all samples samples
    fig, ax = plt.subplots(figsize=(10, 8))

    ind = np.arange(len(values))
    width = 0.8

    plt.bar(ind, values, width, color='lightgreen', edgecolor='black')

    # Define the tick format
    ax.set_xticks(ind)
    ax.set_xticklabels(data_sum.index.values)

    # Add numbers of sampled containing AR genes to the plot
    for i, v in enumerate(values):
        plt.text(i + .0, v * 0.45, str(round(v)), color='black',
                 fontweight='bold', fontsize=14, ha='center', va='center')

    plt.margins(x=0.01, y=0.02)

    # Add title and axis labels
    plt.title('Numbers of Samples Containing Antibiotic Resistance Genes',
              fontsize=16, style="italic")
    plt.xlabel('Antibiotic Resistance Gene')
    plt.ylabel('Number of samples')

    return plt


def plot_gene_correlations(data):
    """Creates a plot for correlation matrix of AB resistance genes.

        Calculates correlation matrix for AB resistance genes detected in teh samples.
        Uses sns.heatmap to visualize the correlation matrix,

        Args:
            A dataframe with data on AB resistance genes detected in the samples.
            Each row is a sample, each column is an AR gene.

        Returns:
            A seaborn heatmap for correlation matrix for AB resistance genes.
    """

    # Calculate Pearson correlation coefficients (correlation matrix)
    # for gene pairs
    gene_corr = data.corr()

    # Plot a bar plot demonstrating presence of antibiotic resistance genes
    # in all samples
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.heatmap(gene_corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlations between Antibiotic Resistance Genes \n" +
              "Detected in the Samples", fontsize=16, style="italic")

    return plt


def plot_multiple_cell_count_histograms(df, bins=20):
    """Creates a set of cell count histograms.

        Creates histograms for each organism cell counts in all samples.
        Generates a matplotlib plot covering all these histograms.

        Args:
            A dataframe with data on cell counts.
            Each row is a sample, each column is an organism.

        Returns:
            A matplotlib histogram plot for organism cell counts in all samples
    """

    # Create subplots for the histograms
    n_col = len(df.columns)
    nrow = 2
    ncol = ceil(n_col / nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 8))
    fig.suptitle('Histograms of Cell Counts For all Organisms', fontsize=24, style="italic")

    for i, ax in enumerate(axes.flatten()):
        # Create a histogram for a single organism cell counts
        ax.hist(np.array(df['Organism_' + str(i + 1)]), bins=bins, density=True,
                alpha=0.8, color='b', edgecolor='black')
        ax.set_ylim([0, 0.001])

        # Set title for a histogram
        ax.set_title('Organism_' + str(i + 1))

    fig.text(.5, -.05, "x-axis - cell counts, y-axis - probability density \n" +
             "The peaks around zero are truncated. They correspond to zero cell counts. " +
             "See next plot for detail.",
             ha='center', fontsize=20)

    plt.tight_layout()

    return plt


def plot_stats_for_zero_cell_counts(data_df_cellcount):
    """Creates a plot of counts of the samples without any cells detected.

        Calculates counts of the samples without any cells detected for each organism type.
        Creates a matplotlib plot for these counts.

        Args:
            A dataframe with data on sample cell counts.
            Each row is a sample, each column is an organism.

        Returns:
            A matplotlib bar chart for counts of the samples without any cells detected.
    """

    # Count numbers of gene detections

    colnames = data_df_cellcount.columns

    zerocounts = np.zeros(len(colnames), dtype=int)

    for i, col in enumerate(colnames):
        zerocounts[i] = data_df_cellcount[col][data_df_cellcount[col] == 0].count()

    values = np.array(zerocounts)

    # Plot a barplot demonstrating presence of antibiotic resistance genes
    # in all samples samples
    fig, ax = plt.subplots(figsize=(10, 8))

    ind = np.arange(len(values))
    width = 0.8

    plt.bar(ind, zerocounts, width, color='lightgreen', edgecolor='black')

    # Define the tick format
    ax.set_xticks(ind)
    ax.set_xticklabels(colnames, rotation=90)

    # Add
    for i, v in enumerate(zerocounts):
        plt.text(i + .0, v * 0.45, str(round(v)), color='black',
                 fontweight='bold', fontsize=14, ha='center', va='center')

    plt.margins(x=0.02, y=0.02)

    # Add title and axis labels
    plt.title('Numbers of Samples With Zero Cell Counts by Organism',
              fontsize=16, style="italic")
    plt.xlabel('Organism name')
    plt.ylabel('Number of samples with zero cell counts by organism')

    return plt


def load_data(datafile_name):
    """Reads a CSV data file from work directory.

            Args:
                A name of the .csv data file located in home directory

            Returns:
                A data frame containing information from the .csv data file.
    """
    data_df = pd.read_csv(datafile_name)

    return data_df


def load_additional_data(main_data, add_data_file):
    """Reads a CSV file with additional data and appends it to already existing data.

        Reads a CSV file with additional data from working directory.
        Appends additional data to already existing dataframe.
        Returns the resulting dataframe.

        Args:
            A dataframe with data already in use.
            File name for the files with additional data.

        Returns:
            A data frame generated by concatenation or existing and additional data.
    """

    add_data = pd.read_csv(add_data_file)

    all_data = pd.concat([main_data, add_data])

    return all_data


def main():
    """Reads a CSV file with urine sample tests, runs data analysis and generates plots for the analysis results.

            Reads a CSV file with urine sample tests.
            Runs several functions to perform data analysis and to generate teh relevant plots:
            plot_ab_resistance - for samples' antibiotic resistance
            plot_gene_counts & plot_gene_correlations - for gene detection
            plot_multiple_cell_count_histograms & plot_stats_for_zero_cell_counts - for cell counts.

            Args:

            Returns:
                A set of plots presenting the results of the data analysis.
        """

    datafile_name = 'urine_test_data.csv'
    data_df = load_data(datafile_name)

    # Split data into three dataframes:
    # antibiotic resistance, cell count and antibiotic resistance (AR) gene detection
    data_df_anti = data_df[[col for col in data_df.columns if "Antibiotic" in col]]
    data_df_cellcount = data_df[[col for col in data_df.columns if "Organism" in col]]
    data_df_genes = data_df[[col for col in data_df.columns if "Gene" in col]]

    # Generating all the plots
    plt = plot_ab_resistance(data_df_anti)
    plt.show()

    plot_gene_counts(data_df_genes)
    plt.show()

    plot_gene_correlations(data_df_genes)
    plt.show()

    plot_multiple_cell_count_histograms(data_df_cellcount)
    plt.show()

    plot_stats_for_zero_cell_counts(data_df_cellcount)
    plt.show()


if __name__ == "__main__":

    main()
