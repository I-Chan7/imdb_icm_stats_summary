"""Provide summary statistics of user data exported from imdb.com and icheckmovies.com"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def initialise_data():
    """Load, merge, tranform data and return result as DataFrame"""
    icm = pd.read_csv('checked.csv', encoding='latin-1')
    imdb = pd.read_csv('ratings.csv', encoding='latin-1')

    # merge icm and imdb on imdburl using inner join
    imdb['URL'] = imdb['URL'].map(lambda x: x.replace('s', ''))
    df = icm.merge(imdb, left_on='imdburl', right_on='URL', how='inner')

    df['checkedcount'] = df['checkedcount'].astype(int)

    # create 'Decade' attribute from 'Year'
    df['Decade'] = df['Year'].map(lambda x: str(x)[:3] + '0s')

    # transform 'IMDb Rating', 'checkedcount' and 'Runtime (mins)' into range attributes
    df['IMDb Rating Range'] = df['IMDb Rating'].map(lambda x: np.nan if pd.isna(x)
                                                    else '8+' if x >= 8
                                                    else '1.0-4.9' if x <= 4.9
                                                    else str(x)[0] + '.0-' + str(x)[0] + '.4'
                                                    if 0 <= int(str(x)[-1]) <= 4
                                                    else str(x)[0] + '.5-' + str(x)[0] + '.9')

    df['iCM Checks Range'] = df['checkedcount'].map(lambda x: '10000+' if x >= 10000
                                                    else '5000-9999' if x >= 5000
                                                    else '1000-4999' if x >= 1000
                                                    else '400-999' if x >= 400
                                                    else '<400')

    df['Runtime Range (mins)'] = df['Runtime (mins)'].map(lambda x: '181+' if x >= 181
                                                                    else '151-180' if 151 <= x <= 180
                                                                    else '121-150' if 121 <= x <= 150
                                                                    else '106-120' if 106 <= x <= 120
                                                                    else '91-105' if 91 <= x <= 105
                                                                    else '76-90' if 76 <= x <= 90
                                                                    else '41-75' if 41 <= x <= 75
                                                                    else '<=40')
    return df


def apply_filters(df, filters, sort_by, sort_ascending):
    """Filter and sort dataframe then return updated dataframe"""
    if filters is not None:
        df = df[filters]
    df = df.sort_values(by=sort_by, ascending=sort_ascending)
    df.set_index(np.arange(len(df))+1, inplace=True)
    return df


def count_non_atomic_fields(df, attribute):
    """Split and aggregate attributes containing non-atomic values(Directors and Genres) and return result"""
    counts = {}

    # store counts and average ratings in dictionary
    def add_to_dict(x):
        if x in counts.keys():
            counts[x]['Average Rating'] = (counts[x]['Average Rating'] * counts[x]['# of Films']
                                         + row['Your Rating']) / (counts[x]['# of Films'] + 1)
            counts[x]['# of Films'] += 1
        else:
            counts[x] = {'Average Rating': row['Your Rating'], '# of Films': 1}

    for index, row in df.iterrows():
        if not pd.isna(row[attribute]):
            if ',' in str(row[attribute]):  # if multiple values found
                split_list = row[attribute].split(', ')
                for i in split_list:
                    add_to_dict(i)
            else:
                add_to_dict(row[attribute])

    # transform dictionary to dataframe amd create '%' attribute from '# of Films'
    result_df = pd.DataFrame.from_dict(counts, orient='index', columns=(['# of Films', 'Average Rating']))
    result_df['Average Rating'] = round(result_df['Average Rating'], 3)
    result_df['(%)'] = result_df['# of Films'].map(lambda x: '('+str(round((x/len(df)*100), 2))+'%)')
    return result_df


def display_results(df, director_df, genre_df, columns_selected):
    """Print and plot results"""
    favourited = len(df[df['favorite'] == 'yes'])
    disliked = len(df[df['disliked'] == 'yes'])
    # set index to counts from 1 to length of dataframe
    df.set_index(np.arange(0, len(df)) + 1, inplace=True)
    print(df[columns_selected].to_string())
    print('\nSeen: {}'.format(len(df)))
    print('Favourited: {} ({:.2f}%)'.format(favourited, favourited / len(df) * 100))
    print('Disliked: {} ({:.2f}%)'.format(disliked, disliked / len(df) * 100))
    print('Number of Unique Directors: {}'.format(len(director_df)))
    print('Total Watch Time: {:.0f} minutes / {:.0f} hours {:.0f} minutes'
          .format(df['Runtime (mins)'].sum(), df['Runtime (mins)'].sum() / 60, df['Runtime (mins)'].sum() % 60))
    print('Longest Runtime: {} ({} mins)'
          .format(df.loc[df['Runtime (mins)'] == df['Runtime (mins)'].max(), 'Title'].to_string(index=False),
                  int(df.loc[df['Runtime (mins)'] == df['Runtime (mins)'].max(), 'Runtime (mins)'])))

    print('\nYour Ratings:')
    user_ratings_count = df['Your Rating'].value_counts().sort_index(ascending=False)
    for index, value in user_ratings_count.items():
        print('{}: {} ({:.2f}%)'.format(index, value, value / len(df) * 100))
    print('Average Rating: {:.5f}\n'.format(df['Your Rating'].mean()))

    # print decade stats
    decades_group = df.groupby('Decade').agg({'Your Rating': ['count', 'mean']})
    decades_group.columns = decades_group.columns.droplevel(0)
    decades_group.columns = ['# of Films', 'Average Rating']
    decades_group['Average Rating'] = round(decades_group['Average Rating'], 3)
    decades_group['(%)'] = decades_group['# of Films'].map(lambda x: '(' + str(round((x / len(df) * 100), 2)) + '%)')
    print(decades_group[['# of Films', '(%)', 'Average Rating']])
    print('\n')

    # print imdb rating stats
    imdb_ratings_group = df.groupby('IMDb Rating Range').agg({'Your Rating': ['count', 'mean']})
    imdb_ratings_group.columns = imdb_ratings_group.columns.droplevel(0)
    imdb_ratings_group.columns = ['# of Films', 'Average Rating']
    imdb_ratings_group['Average Rating'] = round(imdb_ratings_group['Average Rating'], 3)
    imdb_ratings_group['(%)'] = imdb_ratings_group['# of Films']\
        .map(lambda x: '(' + str(round((x / len(df) * 100), 2)) + '%)')
    print(imdb_ratings_group[['# of Films', '(%)', 'Average Rating']].sort_index(ascending=False))
    print('Average IMDb Rating: {:.5f}\n'.format(df['IMDb Rating'].mean()))

    # print runtime stats
    runtime_group = df.groupby('Runtime Range (mins)').agg({'Your Rating': ['count', 'mean']}).reindex(
        ['181+', '151-180', '121-150', '106-120', '91-105', '76-90', '41-75', '<=40'])
    runtime_group.columns = runtime_group.columns.droplevel(0)
    runtime_group.columns = ['# of Films', 'Average Rating']
    runtime_group['Average Rating'] = round(runtime_group['Average Rating'], 3)
    runtime_group['(%)'] = runtime_group['# of Films'].map(lambda x: '(' + str(round((x / len(df) * 100), 2)) + '%)')
    print(runtime_group.loc[~runtime_group['# of Films'].isnull(), ['# of Films', '(%)', 'Average Rating']])
    print('Average Length: {:.0f} minutes\n'.format(df['Runtime (mins)'].mean()))

    # print icm check stats
    icm_checks_group = df.groupby('iCM Checks Range').agg({'Your Rating': ['count', 'mean']}).reindex(
        ['10000+', '5000-9999', '1000-4999', '400-999', '<400'])
    icm_checks_group.columns = icm_checks_group.columns.droplevel(0)
    icm_checks_group.columns = ['# of Films', 'Average Rating']
    icm_checks_group['Average Rating'] = round(icm_checks_group['Average Rating'], 3)
    icm_checks_group['(%)'] = icm_checks_group['# of Films']\
        .map(lambda x: '(' + str(round((x / len(df) * 100), 2)) + '%)')
    print(icm_checks_group.loc[~icm_checks_group['# of Films'].isnull(), ['# of Films', '(%)', 'Average Rating']])
    print('Average # of Check: {:.0f}\n'.format(df['checkedcount'].mean()))

    # print offical list stats
    official_lists_group = df.groupby('officialtoplistcount').agg({'Your Rating': ['count', 'mean']})
    official_lists_group.columns = official_lists_group.columns.droplevel(0)
    official_lists_group.columns = ['# of Films', 'Average Rating']
    official_lists_group['Average Rating'] = round(official_lists_group['Average Rating'], 3)
    official_lists_group['(%)'] = official_lists_group['# of Films']\
        .map(lambda x: '(' + str(round((x / len(df) * 100), 2)) + '%)')
    print(official_lists_group[['# of Films', '(%)', 'Average Rating']].sort_index(ascending=False))

    # print genre stats
    print('\nGenres')
    print(genre_df[['# of Films', '(%)', 'Average Rating']].sort_index())

    # print director stats
    print('\nTop 10 most seen directors:')
    print(director_df[['# of Films']].sort_values('# of Films', ascending=False).head(10))

    # set minimum film count for director ranking(based on average rating)
    director_min_films = 3
    director_min_films_filter = director_df['# of Films'] >= director_min_films
    # print director ranking only if there are directors with # films higher or equal to the specified minimum
    if director_df['# of Films'].max() >= director_min_films:
        print('\nTop 10 directors by average rating (min {} films):'.format(director_min_films))
        print((director_df[['Average Rating']][director_min_films_filter].sort_values('Average Rating', ascending=False)
               .head(10)))

    # plot results
    groups_to_plot = [runtime_group, icm_checks_group]
    xlabels = ['Runtime (mins)', 'iCM Check Count', 'Genres']

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 35))
    plt.subplots_adjust(left=0.06, right=0.94, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
    ax = df['Your Rating'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0])
    ax.tick_params(axis='x', labelrotation=0, labelsize='small')
    ax.set_title('My Ratings')
    ax.set_xlabel('Rating')
    ax.set_ylabel('# of Film')

    ax = df['IMDb Rating Range'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0])
    ax.tick_params(axis='x', labelrotation=45, labelsize=6)
    ax.set_title('IMDb Ratings')
    ax.set_xlabel('Rating')
    ax.set_ylabel('# of Film')

    ax = genre_df['# of Films'].plot(kind='bar', ax=axes[0, 1])
    ax.tick_params(axis='x', labelrotation=90, labelsize=5)
    ax.set_title('Number of Films Seen by Genre')
    ax.set_ylabel('# of Films')
    ax.set_xlabel('Genre')

    ax = genre_df['Average Rating'].plot(kind='bar', ax=axes[1, 1])
    ax.tick_params(axis='x', labelrotation=90, labelsize=5)
    ax.set_title('Average Rating by Genre')
    ax.set_ylabel('Average Rating')
    ax.set_xlabel('Genre')

    for i, item in enumerate(groups_to_plot):
        ax = groups_to_plot[i].loc[::-1, '# of Films'].plot(kind='bar', ax=axes[0, i+2])
        ax.tick_params(axis='x', labelrotation=45, labelsize=6)
        ax.set_title('Number of Films Seen by '+xlabels[i])
        ax.set_ylabel('# of Films')
        ax.set_xlabel(xlabels[i])

        ax = groups_to_plot[i].loc[::-1, 'Average Rating'].plot(kind='bar', ax=axes[1, i+2])
        ax.tick_params(axis='x', labelrotation=45, labelsize=6)
        ax.set_title('Average Rating by ' + xlabels[i])
        ax.set_ylabel('Average Rating')
        ax.set_xlabel(xlabels[i])
    plt.show()


def main():
    df = initialise_data()

    # list of filters
    watched_during = df['checked'].str.contains('2019')
    filmlength = df['Runtime (mins)'] <= 45
    check_count = df['checkedcount'] < 400
    title_type = df['Title Type'] == 'movie'
    user_rating = df['Your Rating'] == 3
    director = df['Directors'].str.contains('Unknown')
    title = df['Title'].str.contains('Frost')
    release_year = df['Year'] < 2018
    imdb_votes = (df['Num Votes'] < 1200) & (df['Num Votes'] > 1000)
    genre = df['Genres'].str.contains('Western')
    imdb_rating = df['IMDb Rating'] > 8

    # set filters to None if no filters to apply
    filters = None
    sort_by = 'Your Rating'
    sort_ascending = False

    # list of columns:
    # 'title', 'year', 'url', 'checkedcount', 'favouritecount', 'officialtoplistcount', 'usertoplistcount',
    # 'akatitle', 'imdburl', 'checked', 'favorite', 'disliked', 'watchlist', 'owned', 'Const','Your Rating',
    # 'Date Rated', 'Title', 'URL', 'Title Type','IMDb Rating', 'Runtime (mins)', 'Year', 'Genres', 'Num Votes',
    # 'Release Date', 'Directors'
    columns_selected = ['Title', 'Year', 'Your Rating', 'IMDb Rating']

    # apply filters when filters is not set to 'None'
    df = apply_filters(df, filters, sort_by, sort_ascending)
    director_df = count_non_atomic_fields(df, 'Directors')
    genre_df = count_non_atomic_fields(df, 'Genres')
    display_results(df, director_df, genre_df, columns_selected)


main()