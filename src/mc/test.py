import inspect
from IPython.display import HTML, display
from . import distributions, experiments, samplings

def get_classes_in_module(module):
    classes = []
    for name, member in inspect.getmembers(module):
        if inspect.isclass(member):
            classes.append(member)
    return classes


def run_all():
    '''
    Iterate all the classes and run with the default params.
    '''
    for module in [distributions, experiments, samplings]:

        all_classes = get_classes_in_module(module)

        # 打印所有类的名称
        for MC in all_classes:
            if MC.__name__ == 'McBase':
                continue

            display(HTML('<h2>' + module.__name__+ '.' + MC.__name__ + str(inspect.signature(MC)) + '</h2>'))
            display(HTML('<pre>' + MC.__doc__ + '</pre>'))
            
            if MC.__init__.__doc__ is not None:
                display(HTML('<pre>' + MC.__init__.__doc__ + '</pre>'))

            obj = MC()
            obj.run()

            display(HTML('<br/><hr/><br/>'))
    