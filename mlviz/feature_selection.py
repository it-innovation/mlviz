# Author ZDZ, refactoring FPC 2019

#Bokeh modules
from bokeh.layouts import column, Row
from bokeh.models import ColumnDataSource, Slider, HoverTool
from bokeh.plotting import figure
from bokeh.themes import Theme
from bokeh.io import show, output_notebook
from bokeh.models.widgets import Panel, Tabs

#SKLearn modules
from sklearn.feature_selection import mutual_info_classif as mi
from sklearn.feature_selection import f_classif as fvalue
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.multiclass import type_of_target
from sklearn.impute import SimpleImputer

#Various sci modules
import math
import numpy as np
import pandas as pd

#Miscellaneous imports
import yaml
from IPython.core.display import Image
import copy

class UVFS():
    """
    Calculates univariate statistical tests to allow feature selection.
    
    Parameters:
    -----------
    X : pd.DataFrame, shape [n_instances, n_features] (default=None)
        Training set.

    y : array-like, shape [n_instances] (default=None)
        Target class of each training instance.
    
    url :str, (default="localhost:8888")
        Location (port) of Jupyter notebook to output
    
    data_dictionary : dict
        An sklearn data dictionary, used to load toy datasets.

    Attributes:
    -----------
    X : pd.DataFrame, shape [n_instances, n_features]
        Training set.

    y : array-like, shape [n_instances]
        Target class of each training instance.

    feature_names : list
        List of feature names selecte by user.
    """

    def __init__(self, X=None, y=None, url="localhost:8888", 
                 data_dictionary=None):
        """
        Constructer for the UVFS tool.
        """
        self.get_data(X=X, y=y, data_dictionary=data_dictionary)
        self._data_dictionary = None
        self._f_value = None
        self._mi = None
        self._source_mi = None
        self._source_f_value = None
        output_notebook()

    def get_data(self, X=None, y=None, data_dictionary=None):
        """
        Loads or prepares the data.

        X : pd.DataFrame, shape [n_instances, n_features] (default=None)
            Training set.

        y : array-like, shape [n_instances] (default=None)
            Target class of each training instance.
    
        data_dictionary : dict, (default=None)
            An sklearn data dictionary, used to load toy datasets.
        """
        if all(i is None for i in [X, data_dictionary]):
            raise Exception('No data provided.')
        if data_dictionary is not None:
            self. X = pd.DataFrame(data_dictionary['data'], 
                                   columns=data_dictionary['feature_names'])
            self.y = data_dictionary['target']
            self.feature_names = data_dictionary['feature_names']
        else:
            self.X, self.y = X, y
            self.feature_names = X.columns.tolist()
        
    def compute_information(self, metrics=["mi", "f-value"]):
        """
        Computes the different statistical tests. Current metrics supported
        are mutual information and f-value.

        Parameters:
        -----------
        metrics : list, shape [n_metrics] (default=["mi", "f-value"])
            The different statistical tests to use.

        TO DO:

        We should not be transforming the data. 

        We don't want to be computing statistics on transformed data in case this isn't what the user wants.
        """
        
        # the metrics currently implemented 
        implemented_metrics = ["mi", "f-value"]
        
        if any(metric not in implemented_metrics for metric in metrics):
                raise ValueError('Unknown metric provided. '
                                 'Available metrics {}'.format(str(metrics)))

        # This shouldn't be done in this function.
        # it should be done in a method which 'cleans' the 
        # input data.
        #Discretise target, if continuous
        target_type = type_of_target(self.y)

        if target_type not in ["binary", "multiclass", "multiclass-multioutput",
                               "multilabel-indicator", "multilabel-sequences"]: 
            print("Discretising ... ", end="")
            
            #Discretise target
            binner = KBinsDiscretizer(n_bins=7,
                                      encode="ordinal",
                                      strategy="uniform")
            binner.fit(y)
            y = binner.transform(self.y)
            print("DONE!")
        else:
            y = self.y
        
        # All this should be done in the "get data"  method.
        #Imput missing values
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputer.fit(self.X) 
        X = imputer.transform(self.X)

        # this needs to be tidied
        #Calculate each metric
        if "f-value" in metrics:
            print("Calculating f-value ... ", end="")
            f, p = fvalue(X, y)
            self._f_value = f
            print("DONE!")

        if "mi" in metrics:
            print("Calculating MI ... ", end="")
            minf = mi(X, y, n_neighbors=7)
            self._mi = minf
            print("DONE!")

    def _f_value_slider_callback(self, attr, old, new):
        """
        Call back for the f value slider.

        Parameters:
            attr (str): source column to monitor
            old (array):
            new (array):
        """
        if new==0:
            selected_idxs = list(range(self.feature_names))
        else:
            mi_all = np.array(self._f_value)
            mi_all = np.nan_to_num(mi_all)
            selected_idxs, = np.nonzero(mi_all >= new)
        self._source_f_value.selected.indices = selected_idxs.tolist()
        self.f_value_value = new

    def _f_value_panel(self):
        """
        Creates the f value panel.

        Returns:
        --------
        panel : bokeh.layouts.Panel object
            Panel containing relevent bokeh models
        """
        if self._f_value is None:
            self.compute_information(["f-value"])

        y_ax = np.array(self._f_value)
        x_ax = self.feature_names
 
        TOOLS = ("box_select, box_zoom, pan, tap, wheel_zoom,"
                " reset, save,zoom_in, zoom_out, crosshair")
                
        self._source_f_value = ColumnDataSource(dict(x=x_ax, y=y_ax))

        plot = figure(x_range=self._source_f_value.data['x'], 
                      tools=TOOLS,
                      active_drag="box_select")

        plot.vbar(source=self._source_f_value, x='x',top='y', width=0.5)
        plot.xaxis.major_label_orientation = math.pi/2

        hover = HoverTool(tooltips=[('Var.', '@x'), ('f-value', '@y')])
        plot.add_tools(hover)

        slider_max_val = np.nanmax(y_ax)
        slider_step = int(slider_max_val/100)
        
        slider = Slider(start=0,
                        end=slider_max_val,
                        value=0,
                        step=slider_step,
                        title="Min. f-value")
        slider.on_change('value', self._f_value_slider_callback)
        panel = Panel(child=column(slider, plot), title='F-value')
        return panel

    def _mi_slider_callback(self, attr, old, new):
        """
        Callback for the mi slider.

        Parameters:
        ----------
        attr : str
            Attribute to monitor (i.e., indices, value) of data.
        
        old : Any
            Old value(s) of attr

        new : Any
            New value(s) of attr.
        """

        # it would be nice if these variable names were more descriptive
        if new == 0:
            selected_idxs = list(range(self.feature_names))
        else:
            mi_all = np.array(self._mi)
            mi_all = np.nan_to_num(mi_all)
            selected_idxs, = np.nonzero(mi_all >= new)
        self._source_mi.selected.indices = selected_idxs.tolist()
        self.mi_value = new

    def _mi_panel(self):
        """
        Creates the mutual information panel.

        Returns:
        --------
        panel : bokeh.layouts.Panel object
            Panel containing relevent bokeh models
        """
        if self._mi is None:
            self.compute_information(['mi'])
        y_ax = np.array(self._mi)
        x_ax = self.feature_names
        TOOLS = ('box_select, box_zoom, pan, tap, wheel_zoom,'
                ' reset, save,zoom_in, zoom_out, crosshair')

        self._source_mi = ColumnDataSource(dict(x=x_ax,y= y_ax))
        
        plot = figure(x_range=self._source_mi.data['x'],
                      tools=TOOLS,
                      active_drag="box_select")
        
        plot.vbar(source = self._source_mi, x='x', top='y', width=0.5)
        plot.xaxis.major_label_orientation = math.pi/2

        hover = HoverTool(tooltips=[('Var.', '@x'), ('MI', '@y')])
        plot.add_tools(hover)

        slider_max_val = np.nanmax(y_ax)
        slider_step = int(slider_max_val/100)
        
        slider = Slider(start=0,
                        end=slider_max_val,
                        value=0,
                        step=slider_step,
                        title="Min. MI")
        slider.on_change("value", self._mi_slider_callback)
        panel = Panel(child=column(slider, plot), title="Mutual Information")
        return panel

    def get_selected_features(self, metric="mi", return_names=True):
        """
        Gets the features selected by the thresholded Mutual information.
        
        Parameters:
        -----------
        metric : str, (default="mi") 
            Metric used to select the features. Possible values are "mi" or 
            "f-value"
        
        return_names : bool, (default=True)
            whether to return name or numerical indices of selected features.
        
        Returns:
        --------
        selected_features : array-like
            Array of the features selected by user.
        """
        sources = {"mi":self._source_mi,
                   "f-value":self._source_f_value}
        print(f"Getting features selected by {metric}")
        source = sources[metric]
        if source is None:
            raise Exception('No source available.')
        selected_idxs = sorted(source.selected.indices)
        if len(selected_idxs)==0:
            selected_idxs = list(range(0, len(self.feature_names)))
        if return_names is False:
            selected_features = selected_idxs
        else:
            selected_features = [self.feature_names[i] for i in selected_idxs]
        return selected_features

    def _notebook_ui(self, doc):
        """
        Shows and sets up the user interface.
        
        Parameters:
        -----------
        doc : bokeh.Document
            Document instance to collect bokeh models in.
        """
        # this throws a depreciation warning.
        doc.theme = Theme(json=yaml.load("""
            attrs:
                Figure:
                    background_fill_color: "#DDDDDD"
                    outline_line_color: white
                    toolbar_location: above
                    height: 500
                    width: 800
                Grid:
                    grid_line_dash: [6, 4]
                    grid_line_color: white
        """)) 

        tab_fc = self._f_value_panel()
        tab_mi = self._mi_panel()
        tabs = Tabs(tabs=[tab_fc, tab_mi])
        doc.add_root(tabs)

    @staticmethod
    def show_example(fpath='data/static/UVFS_example.gif'):
        """
        Displays an interactive example of the UVFS tool usage.

        Parameters:
        ----------
        fpath : str, (default="data/static/UVFS_example.gif")
            fpath of .gif to display when called.

        Returns
        -------
        image, Ipython.core.display.Image instance
            .gif to be displayed in notebook.
        """
        return Image(filename=fpath)

    def run_ui(self):
        """
        Shows the UI.
        """
        return show(self._notebook_ui)