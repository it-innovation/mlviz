# Author FPC, 2019

import pandas as pd
import numpy as np

from utilities import utils

from bokeh.plotting import figure
from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets.buttons import Button
from bokeh.models.widgets import Select, Slider
from bokeh.models.glyphs import Text

from bokeh.layouts import (layout, 
                           Spacer, 
                           Row, 
                           Column, 
                           gridplot)
from bokeh.io import show

from bokeh.application.handlers.function import FunctionHandler
from bokeh.application import Application

# things for controlling colour
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256, Category20, inferno
from bokeh.models import ColorBar, BasicTicker

# dimensionality reduction algorithms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

from IPython.core.display import Image

class DraughtPlot():
    """
    Plots a draughtsman plot of provided data.

    Attributes:
        X (pd.DataFrame): (n x m) dataframe with n instances and m      
                          features.
        y (np.ndarray): 1D array of target values for each of the 
                            n instances in X.
        brush_idxs (np.ndarray): idxs of different clusters to plot.

        features (list): list (of max length 10) of features to plot.
    """

    def __init__(self, X, y=None, features=None, url='localhost:8888'):
        """
        Constructer for the DraughtPlot class. 
        
        Parameters:
            X (pd.DataFrame): (n x m) dataframe with n instances and m      
                              features. 
            y (np.ndarray): 1D array of target values for each of the 
                            n instances in X. Defaults to None.
            features (list): list (of max length 10) of features to plot.
                             Defaults to None. If you want to plot the
                             target include a list of features with 
                             'target' in.
            url (str): url of notebook for output.
        """
        self.X = X
        self.y = y
        self.features = features
        # could be updated to make an educated guess of the five features.
        if self.features is None:
            self.features = self.X.columns[:5].tolist()
        if 'target' in self.features:
                self.X['target'] = self.y
        self.corr_matrix = X[self.features]
        app = Application(FunctionHandler(self._make_bokeh_doc)) # make document
        show(app, notebook_url=url) # show document in notebook

    def _make_bokeh_doc(self, doc):
        """
        Function which is called when generating the Bokeh document. Main 
        function which constructs the application.
        
        Parameters:
            doc (bokeh.Document): Document instance to collect bokeh models in.
        """
        self._make_data_source()
        self._make_figures()
        self._initalise_figures()
        self._make_widgets()
        doc_layout = self._make_doc_layout()
        doc.add_root(doc_layout)

    def _make_doc_layout(self):
        """
        Constructs the document layout object.
        
        Returns:
            doc_layout (layout): returns grid layout of bokeh layout objects.
        """
        cb_figure = Column(self._create_colour_bar())
        widgets = Column(self.widgets['select_color'],
                         self.widgets['change_color'])
        draughtsman_plot = gridplot(self.figures.tolist())
        doc_layout = layout([[draughtsman_plot, cb_figure, widgets]])
        return doc_layout
    
    def _make_figures(self, fig_props={'plot_height':150, 
                                       'plot_width':150, 
                                       'tools':('box_zoom,'
                                                'wheel_zoom,'
                                                'box_select,'
                                                'lasso_select,'
                                                'reset,'
                                                'help,'
                                                'save')}
                                       ):
        """
        Creates a grid of empty figures.

        References to the figures are stored in a numpy array, the value
        of each entry is the figure object.
        """
        self.figures = np.array([])
        # could replace this with a single for loop using itertools
        for i, __ in enumerate(self.features):
            for j, __ in enumerate(self.features):
                xlabel, ylabel = self._get_axis_labels(i, j)
                fig = figure(x_axis_label=xlabel,
                             y_axis_label=ylabel,
                             **fig_props)
                self.figures = np.append(self.figures, fig)
        num_features = len(self.features)
        self.figures = self.figures.reshape((num_features, num_features))

    def _make_widgets(self, widget_width=200):
        """
        Generates the widgets the user can use to interact with the graph.
        """
        # dictionary storing: (widget type, kwargs, callback) for each widget
        widget_definitions = {'select_color':(Select,
                                              {'options':self.X.columns.tolist(),
                                               'title':'Select colour',
                                               'width':widget_width},
                                                None
                                             ),
                              'change_color':(Button,
                                              {'label': 'Change colour',
                                               'width': widget_width},
                                              self._change_color,
                                             )
                             }
        self.widgets = {}
        for widget_name, widget_def in widget_definitions.items():
            widget = widget_def[0](**widget_def[1])
            if widget_def[2] is not None:
                widget.on_click(widget_def[2])
            self.widgets[widget_name] = widget

    def _change_color(self):
        """
        Callback for change_color button. 
        """
        feature_name = self.widgets['select_color'].value
        feature_data = self.X[feature_name].values
        self.source.patch({'cluster_number': [(slice(len(feature_data)),
                                               feature_data)]})
        # update the color_mapper limits
        self.color_mapper['transform'].low=feature_data.min()
        self.color_mapper['transform'].high=feature_data.max()
    
    def _get_axis_labels(self, col_num, row_num):
        """
        For give column and row numbers provides the labels for the 
        x and y axis for this figure panel.

        Parameters:
            col_num (int): column index of figure panel
            row_num (int): row index of figure panel

        Returns:
            x_label (str): Label for x axis
            y_label (str): label for y axis
        """
        x_label, y_label = '', ''
        if col_num==(len(self.features)-1):
            x_label = self.features[row_num]
        if row_num==0:
            y_label = self.features[col_num]
        return x_label, y_label

    def _initalise_figures(self):
        """
        For each figure in self.figures plots the relevant graph (text,
        histogram, or scatter) to the figure and stores the GlyphRenderer
        object returned.
        """
        self._glyphs = np.array([]) # for storing GlyphRenders
        self._make_color_mapper()
        for i, fy in enumerate(self.features):
            for j, fx in enumerate(self.features):
                plot_type = self._check_fig_type(i,j)
                if plot_type=='histogram':
                    cnts, edges = self._gen_hist_data(self.X[fx].to_numpy())
                    glyphs = self._plot_histogram(figure=self.figures[i,j],
                                                  counts=cnts,
                                                  bin_edges=edges)
                if plot_type=='scatter':
                    glyphs = self._plot_scatter(self.figures[i,j],
                                                fx, 
                                                fy)
                if plot_type=='text':
                    # plot a scatter for the time being
                    glyphs = self._plot_text(self.figures[i,j],
                                                i, 
                                                j)
                    # turn of the figure axis and grid
                    self.figures[i,j].axis.visible = False
                    self.figures[i,j].grid.visible = False
                self._glyphs = np.append(self._glyphs, glyphs)
        num_features = len(self.features)
        self._glyphs = self._glyphs.reshape((num_features, num_features))
        (self._glyphs[1,0].data_source
                          .selected
                          .on_change('indices', self._update_plot))
 
    def _plot_histogram(self,
                        figure, 
                        counts, 
                        bin_edges, 
                        hist_props={'color':'grey',
                                    'selection_color':'grey',
                                    'selection_fill_alpha':0.1,
                                    'fill_alpha':0.4,
                                    'line_alpha':0.75}):
        """
        Plots a histogram to a given figure.

        Parameters:
            figure (bokeh.plotting.figure): figure object to plot too.
            counts (np.ndarray): Counts in each bin of histogram.
            bin_edges (np.ndarray): edges of histogram bins.
            hist_props (dict): kwargs to pass to fig.quad.

        Returns:
            hist (GlyphRenderer): Bokeh GlyphRenderer object of plotted data.
        """
        left_edges = bin_edges[:-1]
        right_edges = bin_edges[1:]
        hist = figure.quad(bottom=0,
                           left=left_edges,
                           right=right_edges,
                           top=counts,
                           **hist_props)
        return hist

    def _make_color_mapper(self):
        """
        Makes the color mapper object used to control the colour of data points 
        in all the plots.
        """
        high_value = np.max(self.source.data['cluster_number'])
        self.color_mapper = linear_cmap(field_name='cluster_number',
                                        palette=Viridis256,
                                        low=1,
                                        high=high_value)

    def _plot_scatter(self, figure, f1, f2):
        """
        Plots a scatter of two features (f1 and f2) for all training instances
        in self.source.

        Parameters:
            figure (bokeh.figure): bokeh figure to plot too
            f1 (str): name of feature to plot on x-axis
            f2 (str): name of feature to plot on y-axis

        Returns:
            scatter (bokeh.GlyphRenderer) GlpyhRenderer object for plotted data.
        """
        scatter = figure.scatter(x=f1, y=f2,
                                 source=self.source,
                                 color=self.color_mapper,
                                 marker='markers',
                                 line_color='black',
                                 line_width=0.3,
                                 nonselection_fill_color=self.color_mapper,
                                 nonselection_fill_alpha=0.1)
        return scatter

    def _create_colour_bar(self):
        """
        Creates a colorbar in an empty bokeh figure.

        Returns:
            colorbar_fig (bokeh.figure): figure containing the colorbar.
        """
        self.color_bar = ColorBar(color_mapper=self.color_mapper['transform'],
                                  width=8)
        colorbar_fig = figure(plot_height=188*len(self.features),
                              plot_width=85,
                              toolbar_location=None,
                              outline_line_color=None)
        colorbar_fig.add_layout(self.color_bar, 'center')
        return colorbar_fig

    def _plot_text(self, figure, col_num, row_num):
        """
        Adds text, describing statistical properties of features of figure
        located at indices (col_num, row_num).
        
        Parameters:
            figure (bokeh.figure): figure object to plot text too
            col_num (int): vertical index of figure panel
            row_num (int): horizontal index of figure panel
        """
        figure_text = self._generate_panel_text(col_num, row_num)
        text = figure.text(x=[0], y=[0],
                           text=[figure_text],
                           text_font='times',
                           text_font_style='italic',
                           text_font_size='12pt',
                           text_align='center',
                           text_baseline='middle')
        return text

    def _generate_panel_text(self, col_num, row_num):
        """
        Returns the text for the figure at (col_num, row_num).

        Parameters:
            col_num (int): vertical index of figure panel
            row_num (int): horizontal index of figure panel

        Returns: 
            text (str): text describing two features correlations.
        """
        correlation_coeff = self.corr_matrix.iloc[row_num, col_num]
        text = 'r: {:.2f}'.format(correlation_coeff)
        return text

    @property
    def corr_matrix(self):
        """
        Get or calculate (set) the correlation matrix of data. The correlation
        matrix is initialised using all the data provided to DraughtPlot.

        To calculate the correlation matrix, the setter calls the .corr()
        method of pd.DataFrame.
        """
        return self._corr_matrix

    @corr_matrix.setter
    def corr_matrix(self, df):
        self._corr_matrix = df.corr()
    
    def _gen_hist_data(self, x, bin_count=20):
        """
        Creates a histogram of provided data (x).

        Parameters:
            x (np.array): 1D array of data to generate histogram of.
            bin_count (int): number of bins in histogram.

        TO DO: 
            1. Need to make bins never change for a given feature.
            Or get them from the figure?
        """
        counts, bins = np.histogram(x, bins=bin_count)
        return counts, bins

    @staticmethod
    def _check_fig_type(col_num, row_num):
        """
        Given the (row, column) index of a figure return the type of figure it
        is. Possible values are 'histogram', 'scatter' or 'text'.

        Parameters:
            col_num (int): vertical index of figure panel
            row_num (int): horizontal index of figure panel

        Returns:
            figure_type (str): type of figure at (col_num, row_num)
        """
        if row_num==col_num:
            figure_type = 'histogram'
        elif col_num>row_num:
            figure_type = 'scatter'
        else:
            figure_type = 'text'
        return figure_type

    def _make_data_source(self):
        """
        Creates a ColumnDataSource of the provided data using given features, 
        ensures a cluster_number column exists in the df.
        """
        try:
            cols_to_keep = self.features + ['cluster_number']
            self.source = ColumnDataSource(self.X[cols_to_keep])
        except KeyError:
            # No cluster_number column defined, make a dummy one
            self.source = ColumnDataSource(self.X[self.features])
            cluster_numbers = np.ones(self.X.shape[0])
            self.source.add(data=cluster_numbers,        
                            name='cluster_number')
        # create markers for the scatter plot if target is binary
        self.markers = np.array(['circle']
                                *len(self.source.data['cluster_number']))
        if utils.is_binary(self.y):
            self.markers[self.y==0] = 'square'
        self.source.add(data=self.markers, name='markers')

    
    def _update_plot(self, attr, old, new):
        """
        Updates the text description with data points select in the scatter.

        Parameters:
            attr (str): attribute to monitor (i.e., indices of selected data)
            old (): old value(s) of attribute.
            new (): new value(s) of attribute.
        """
        inds = new
        # get data selected by user, or all data.
        if len(inds)>0 and len(inds)<self.X.shape[0]:
            X_view = self.X.iloc[inds,:][self.features]
        else:
            X_view = self.X[self.features]
        self.corr_matrix = X_view
        for i, __ in enumerate(self.features):
            for j, __ in enumerate(self.features):
                plot_type = self._check_fig_type(i,j)
                if plot_type=='text':
                    (self._glyphs[i,j]
                        .data_source
                        .data['text']) = [self._generate_panel_text(i,j)]

    @staticmethod
    def show_example(fpath='data/static/DraughtPlot_example.gif'):
        """
        Displays a .gif demonstrating how to use the tool.
        """
        return Image(filename=fpath)

class HistView():
    """
    Histogram viewer which plots histograms of a feature, the feature displayed
    can be controlled by using a slider. Currently all features values must be 
    bounded between 0 and 1.

    Attributes:
        X (pd.DataFrame): (n x m) dataframe with n instances and m      
                          features.
        y (np.ndarray): 1D array of target values for each of the 
                            n instances in X.
    """
    # TO DO: 
    # 1. Add catch / check for if a target is not provided
    # 2. Make robust to non-scaled features.

    def __init__(self, X, y=None, url='localhost:8888', bin_count=50):
        """
        Constructer for the HistView class.

        Parameters:
        X (pd.DataFrame): (n x m) dataframe with n instances and m      
                          features.
        y (np.ndarray): 1D array of target values for each of the 
                            n instances in X. Defaults to None.
        url (str): url of notebook for output.
        bin_count (int): number of bins to seperate the histogram into.
        """
        # Need to add checks to only plot a fraction of the data if it is large.
        self.X, self.y = X, y
        self.bin_edges = np.linspace(0, 1, bin_count+1)
        self._are_features_bounded(X)
        app = Application(FunctionHandler(self._make_bokeh_doc)) # make document
        show(app, notebook_url=url) # show document in notebook

    def _make_bokeh_doc(self, doc):
        """
        Main method controlling the construction of the bokeh document.

        Parameters:
            doc : bokeh Document instance to contain all required Bokeh models.
        """
        self._init_figure()
        self._init_slider()
        self._init_histogram()

        doc_layout = layout([self.slider, self.figure])
        doc.add_root(doc_layout)

    def _are_features_bounded(self, X):
        """
        Checks features are bounded between 0 and 1 and if not raises an 
        exception suggesting the use of MinMaxScaler.

        Parameters:
            X (pd.DataFrame): (n x m) dataframe with n instances and m      
                          features.
        """
        max_feature_value = X.max(axis=0).values
        min_feature_value = X.min(axis=0).values
        features_bounded = all((min_feature_value>-1e-6) &
                               (max_feature_value<1.00001))
        if not features_bounded:
            raise Exception('Features must be bounded between 0 and 1!'
                            'Try using MinMaxScaler from sklearn to prepare'
                            'your features for HistView.')

    def _init_figure(self, fig_props={'plot_height':500, 
                                      'plot_width':500, 
                                      'tools':('box_zoom,'
                                               'wheel_zoom,'
                                               'reset,'
                                               'help,'
                                               'save')}):
        """
        Initiaties the empty figure.

        Parameters:
            fig_props: dictionary of kwargs to be passed to figure object.
        """ 
        self.figure = figure(x_axis_label='Value',
                             y_axis_label='Count',
                             title='Feature Histogram by Class',
                             **fig_props)

    def _init_slider(self):
        """
        Inititiates the feature selection slider.
        """
        self.slider = Slider(start=0, 
                             end=float(self.X.shape[1]-1), 
                             step=1.0,
                             value=0,
                             show_value=False)
        self.slider.on_change('value', self._slider_callback)
        self.slider.title = self.X.columns[0]

    def _slider_callback(self, attr, old, new):
        """
        Callback for the feature selection slider. Code will be ran when the value of the slider changes.
        
        Parameters:
            attr : attribute of slider to monitor
            old : old value of attr
            new : new value of attr
        """
        idx = int(new)
        feature_values = self.X.iloc[:, idx].values
        self._update_histogram(feature_values)
        self.slider.title = self.X.columns[idx]

    def _get_histogram_values(self, values):
        """
        Given an array of values generates a histogram for each value of target.
        These are then stacked in columns.

        Parameters:
            values : array-like object of values to bin.

        Returns:
            count_arr: the counts for each value of target in each bin.
        """
        target_values = np.unique(self.y)
        # init empty array to store counts of each target [n_bins, n_targets]
        count_arr = np.zeros((len(self.bin_edges)-1, len(target_values)))
        for idx, target in enumerate(target_values):
            masked_values = values[self.y==target]
            counts, _ = np.histogram(masked_values, bins=self.bin_edges)
            count_arr[:, idx] = counts
        return count_arr

    def _update_histogram(self, values):
        """
        Function which updates histogram after the slider has changed.

        Parameters:
            values : array-like object of values to bin.
        """
        count_arr = self._get_histogram_values(values)
        hist_bottoms = np.zeros(count_arr.shape[0]) 
        for i, hist in enumerate(self.hists):
            hist.data_source.data['bottom'] = hist_bottoms
            hist.data_source.data['top'] = hist_bottoms+count_arr[:,i]
            hist_bottoms = hist_bottoms + count_arr[:,i]

    def _init_histogram(self):
        """
        Plots the initial histogram, uses the 0th feature.
        """       
        feature_values = self.X.iloc[:,0].values
        count_arr = self._get_histogram_values(feature_values)
        possible_targets = np.unique(self.y)
        num_targets = len(possible_targets)
        try:
            colours = Category20[num_targets]
        except KeyError:
            # if there are two features, use three feature cs
            colours = Category20[num_targets+1]
        
        left_edges = self.bin_edges[:-1]
        right_edges = self.bin_edges[1:]
        hist_bottoms = np.zeros(count_arr.shape[0])
        self.hists = []
        for i in range(num_targets):
            hist = self.figure.quad(bottom=hist_bottoms,
                                    left=left_edges,
                                    right=right_edges,
                                    top=count_arr[:,i]+hist_bottoms, 
                                    alpha=0.75,
                                    color=colours[i],
                                    legend=str(possible_targets[i]))
            hist_bottoms = hist_bottoms + count_arr[:,i]
            self.hists.append(hist)