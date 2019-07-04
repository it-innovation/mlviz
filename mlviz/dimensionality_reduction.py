# Author FPC 2019

import pandas as pd
import numpy as np

from utilities import utils

# libraries for show_example
from IPython.core.display import Image

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

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
from bokeh.palettes import Viridis256, Category20
from bokeh.models import ColorBar, BasicTicker

# dimensionality reduction algorithms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

class HDVis():
    """
    Plots a 2D representation of a high dimensional dataset and allows users to
    interactively extract subsets of the data.
    
    Parameters:
    -----------
    X : pd.DataFrame, shape [n_instances, n_features]
        The training set to transform.

    y : array-like, shape [n_instances]
        Target class of each instance in X.

    url : str, (default="localhost:8888")
        Location (port) of the jupyter notebook to display the figure in.
    
    Additional kwargs are passed to the dimensionality reduction method used.

    Attributes:
    -----------
    X : pd.DataFrame, shape [n_instances, n_features]
        The training set to transform.

    y : array-like, shape [n_instances]
        Target class of each instance in X.
        
    brushes : array-like, shape [n_brushes]
        Array containing the indices of the n_brushes of subsets of the data
        brushed by the users.
    """
    name = "HDVis"
    
    def __init__(self, X, y=None, url="localhost:8888", **kwargs):
        """
        Constructor for the HDVis class.
        """
        self.X, self.y = X, y
        self.features = X.columns.to_numpy()
        self.brushes = []
        self._kwargs = kwargs
        app = Application(FunctionHandler(self._make_bokeh_doc)) # make document
        show(app, notebook_url=url) # show document in notebook
        
    def _make_bokeh_doc(self,
                        doc,
                        fig_props={'plot_height':750,
                                   'plot_width':750,
                                   'tools':('pan,box_zoom,wheel_zoom,'
                                   'reset,box_select,lasso_select,help')}):
        """
        Function which is called when generating the Bokeh document. Main 
        function which constructs the application.
        
        Parameters:
        -----------
        doc : bokeh.Document
            Document instance to collect bokeh models in
        fig_props : dict
            kwargs to pass bokeh.figure instance.

        Returns:
        --------
        doc : bokeh.Document
            The modified document instance.
        """
        self.fig = figure(**fig_props) # create empty central panel
        self._generate_widgets()
        doc_layout = self._make_doc_layout()
        doc.add_root(doc_layout)
        return doc

    @staticmethod
    def _make_data_source(df):
        """
        Creates a ColumnDataSource instance from provided dataframe.

        Parameters:
        -----------
        df : pd.Dataframe, shape [n_rows, n_columns]
            Dataframe to construct ColumnDataSource from.

        Returns:
        --------
        source : ColumnDataSource
            ColumnDataSource instance of the data in df.
        """
        source = ColumnDataSource(df)
        # add a dummy column for colour of each d.p
        source.add(np.zeros(df.shape[0]), name='color')
        return source 

    def _generate_widgets(self):
        """
        Generates the each widget instance used for user interaction.
        
        # TO DO: 
            refactor: change widgets to dict and tuple?
        """
        # generate options for the color dropdown menu
        color_options = ['None', 'Target', *self.features]
        if self.y is None:
            color_options.remove('Target')

        # initiate widget properties: (type, kwargs, callback function)
        self.widget_names = ['method', 'num_train_inst', 'color','change_color',                     'start', 'add', 'clear']
        self.widget_types = {'method':Select,
                             'num_train_inst':Slider,
                             'color':Select,
                             'change_color':Button,
                             'start':Button,
                             'add':Button,
                             'clear':Button
                            }
        self.widget_kwargs = {'method':{'title':'Method:',
                                        'width':259,
                                        'value':'TSNE',
                                        'options':['TSNE', 'PCA', 'UMAP']},
                              'num_train_inst':{'title':'# of data points',
                                                'width':250,
                                                'value':int(self.X.shape[0]/2),
                                                'start':50,
                                                'end':self.X.shape[0],
                                                'step':50},
                              'color':{'width':250,
                                       'title':'Select colour:',
                                       'options':color_options},
                              'change_color':{'width':250,
                                              'label':'Change colour'},
                              'start':{'label':'Run!', 'width':250},
                              'add': {'label':'Add selection',
                                      'width':150},
                              'clear':{'label':'Clear selection',
                                       'width':150}
                             }
        self.widget_callbacks = {'method':None,
                                 'num_train_inst':None,
                                 'color':None,
                                 'change_color':self._color_button,
                                 'start':self._start_button,
                                 'add':self._add_button,
                                 'clear':self._clear_button
                                }
        # initiate the widgets
        self.widgets = {}
        for widget_name in self.widget_names:
            widget = (
                      self.widget_types[widget_name]
                     (**self.widget_kwargs[widget_name])
                     )
            self.widgets[widget_name] = widget
        # add callback to start widget
        self.widgets['start'].on_click(self._start_button)
        
    def _make_doc_layout(self):
        """
        Constructs the document layout object.
        
        Note: Row and Column are experimental features. The positions of 
        the widgets could be hardcoded.
        
        Returns:
        --------
        doc_layout : bokeh.layouts.layout object
            Grid layout of bokeh models.
        """
        top_row = Row(self.widgets['add'],
                      Spacer(width=25),
                      self.widgets['clear'])
        right_column = Column(self.widgets['method'],
                              self.widgets['num_train_inst'],
                              self.widgets['color'],
                              self.widgets['change_color'],
                              self.widgets['start'])
        doc_layout = layout([top_row,
                             [self.fig, right_column]])
        return doc_layout
        
    def _color_button(self):
        """
        Callback for the color button, colors instances by the selected
        feature.

        # TO DO:
            - Refactoring to improve the colouring by target or None.
        """
        try:
            feature_name = self.widgets['color'].value
            if feature_name!='Target': 
                feature_data = self.source.data[feature_name]
            else:
                # get targets for displayed X data
                feature_data = self.y[self.source.data['index']]
            self.source.patch({'color': [(slice(len(feature_data)),
                                          feature_data)]})
            self.color_mapper['transform'].low=feature_data.min()
            self.color_mapper['transform'].high=feature_data.max()
        except:
            # no data points, catches all errors without re-colouring. 
            pass

    def _start_button(self):
        """
        Callback for the run button, performs dimensionality reduction
        and plots the result.

        # TO DO:
            Clear plot / update when start button if pressed for a second time.
        """
        self.widgets['start'].disabled = True
        self._dim_reduction()
        self._plot_embedded_representation()
        
            
    def _add_button(self):
        """
        Callback for the add selections button, appends indices of brushed
        data to self.brushes.
        """
        try:
            brushed_idxs = np.array(self.scatter.data_source.selected.indices)
            # convert brushed indices to those of original df
            original_idxs = self.source.data['index'][brushed_idxs]
            self.brushes.append(original_idxs)
        except:
            print('No data selected.')
        
    def _clear_button(self):
        """
        Callback for the clear selections button, clears self.brushes.
        """
        self.brushes = []
        
    def _activate_widgets(self, widgets_to_activate=['add',
                                                     'clear',
                                                     'change_color']):
        """
        Activates named widgets by assigning callbacks.
        
        Parameters:
        -----------
        widgets_to_activate : list, (default=["add", "clear", "change_color"])
            Widgets to assign callbacks to.
        """
        for widget_name in widgets_to_activate:
            (self.widgets[widget_name]
                 .on_click(self.widget_callbacks[widget_name])
            )
        
    def _dim_reduction(self, num_dim=2):
        """
        Performs the dimensionality reduction, embedding into num_dim
        dimensions.
            
        Parameters:
        -----------
        num_dim : int, (default=2)
            Number of dimensions to embed the data into. Only num_dim=2 is 
            currently supported.
        """
        reduction_methods = {'PCA':PCA, 'TSNE':TSNE, 'UMAP':UMAP}
        # sample the training array
        X_sample = self.X.sample(n=self.widgets['num_train_inst'].value)
        # get the dimensionality reduction method and instantiate it
        dr_method = reduction_methods[self.widgets['method'].value]
        dr_transformer = dr_method(n_components=num_dim, **self._kwargs)
        X_embedded = dr_transformer.fit_transform(X_sample.to_numpy())
        self.source = self._make_data_source(X_sample)
        # append position of training instance in embedded space to data source
        for label, col in zip(['embedded_x','embedded_y'],
                              [X_embedded[:,0], X_embedded[:,1]]):
           self.source.add(data=col, name=label)
            
    def _plot_embedded_representation(self):
        """
        Plots the embedded representation of the training set.
        """
        self.color_mapper = linear_cmap(field_name='color', 
                                        palette=Viridis256, 
                                        low=0, 
                                        high=1)
        self.scatter = self.fig.scatter(x='embedded_x', y='embedded_y',
                                        source=self.source,
                                        color=self.color_mapper,
                                        nonselection_fill_color='grey')
        # add colorbar to plot
        color_bar = ColorBar(color_mapper=self.color_mapper['transform'], 
                             width=8)
        self.fig.add_layout(color_bar, 'right')
        # Activate buttons which require plotted data.
        self._activate_widgets()

    def get_brushed_data(self):
        """
        Returns Dataframe of the brushed data where the clusters are 
        assigned a number in the order they were brushed.

        Returns:
        --------
        brushed_df : pd.Dataframe
            The data brushed by the user.
        """
        try: 
            # get the brushed data and assign a cluster number
            all_brushed_data = np.concatenate(self.brushes).ravel()
            brushed_df = self.X.loc[all_brushed_data,:]
            brushed_df['cluster_number'] = np.zeros(brushed_df.shape[0])
            for i, idxs in enumerate(self.brushes):
                brushed_df.loc[idxs,'cluster_number']= i+1
            if self.y is None:
                brushed_target = None
            else:
                # this isn't completely robust due to mixed pd / np indexing 
                # methods
                brushed_target = self.y[all_brushed_data]
        except IndexError:
            pass # no brushed data
        return brushed_df, brushed_target

    @staticmethod
    def show_example(fpath="data/static/HDViz_example.gif"):
        """
        Displays a .gif demonstrating how to use the tool.

        Parameters:
        -----------
        fpath : str, (default="data/static/HDViz_example.gif")

        Return:
        -------
        Image, Image object
            The .gif to be displayed in a jupyter notebook.
        """
        # this currently uses the .gif within the examples directory
        # it will not work in a generic notebook.
        return Image(filename=fpath)