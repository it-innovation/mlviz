"""Base interactor class for the mlviz library.

This is an experimental branch where we are refactoring the current classes to 
inherent from a parent 'Interactor class'. This will help all tools being built in a consistent way, with the same definitions, variables names. 

It will enable future tools to be built much more quickly by subclassing the Interactor class.
"""
# functions for handling document creation
from bokeh.application.handlers.function import FunctionHandler
from bokeh.application import Application
from bokeh.io import show


from IPython.core.display import Image

class Interactor():
    """
    Interactor class for all tools in MLviz.

    Parameters:
    -----------
    name : str
        The name of the subclass for the given tool (e.g., "HistView")
    
    init_doc : callable
        Function which constructs the bokeh models and returns a layout 
        object.

    url : str, (default="localhost:8888")
        Location (port) of the Jupyter notebook to output document too.

    Attributes:
    -----------
    _widgets : {None or dict}
        Dictionary of tuples of the widgets for a given tool. It is of the form:
        {"widget_name" : (widget_type, widget_kwargs, widget_callback), ...}

    Examples:
    ---------
    The init_doc function should be consistent with:

        def init_doc(self, x, y):

            fig = bokeh.plotting.figure()
            _ = fig.scatter(x, y)
            # ... do other things to figure
            doc_layout = bokeh.layouts.layout([fig])
        
            return doc_layout
    """
    
    def __init__(self, name, init_doc, url="localhost:8888"):
        self.name = name # class name of the Interactor instance
        self.url = url
        self._widgets = None
        self._init_doc = init_doc
        app = Application(FunctionHandler(self._make_doc))
        show(app, notebook_url=self.url)


    def _make_doc(self, doc):
        """
        Constructor which makes the bokeh document.
        
        Parameters:
        -----------
        doc : bokeh.Document instance
            Document to modify and store all Bokeh models.

        Returns:
        --------
        doc : bokeh.Document instance
            The modified Document instance.

        """
        doc_layout = self._init_doc()
        doc.add_layout(doc_layout)
        return doc

    def show_example(self, fpath=None):
        """
        Displays a .gif demonstrating how to use the tool.

        Parameters:
        -----------
        fpath : str, default=None
            Path to the .gif to display. If fpath is None the following fpath
            is used: f"data/static/{self.tool_name}_example.gif"

        Return:
        -------
        Image, Image object
            The .gif to be displayed in a jupyter notebook.
        """
        # this currently uses the .gif within the examples directory
        # it will not work in a generic notebook.
        if fpath is None:
            fpath = "data/static/{}_example.gif".format(self.tool_name)
        return Image(filename=fpath)
