import xarray as xr
from camps.variables import Variable
from camps.names import name_from_var_and_scheme
import camps

@xr.register_dataset_accessor("camps")
class CampsDataset:
    ''' extend xarray Dataset functionality '''

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def var_name(self, var):
        if isinstance(var, camps.Variable):
            print(f'producing variable name based on variable name scheme on dataset: {var.name}')
            return name_from_var_and_scheme(var, None)  #  change None to be the scheme on the file once implemented
            # create Variable name from Dataset name scheme

    def __contains__(self, obj):
        if isinstance(obj, camps.Variable):
            name = self.var_name(obj)
            return name in self._obj
        else:
            return obj in self._obj

    def __getitem__(self, var) -> xr.DataArray:
        if isinstance(var, camps.Variable):
            name = self.var_name(var)
            return self._obj[name].camps.select(var)
        else:
            return self._obj[var]

@xr.register_dataarray_accessor("camps")
class CampsDataarray:
    ''' extend xarray DataArray functionality '''

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def update_meta_attrs(self, var):
        ''' conform metadata to Variable '''
        # The Variable class will define the relevant metadata attrs and values
        # populate self._obj.attrs with the relevent metadata here
        pass

    def select(self, var):
        ''' Return selection of variable from DataArray as DataArray '''
        a = filters[0](self._obj, var)
        for filter in filters[1:]:
            a = filter(a, var)
        return a

filters = list()

def filter(f):
    filters.append(f)
    return f

@filter
def reference_time(data: xr.DataArray, var: camps.Variable) -> xr.DataArray:
    if var.reference_time is None:
        return data
    else:
        return data.sel(reference_time=var.reference_time)

@filter
def time(data: xr.DataArray, var: camps.Variable) -> xr.DataArray:
    if var.time is None:
        return data
    else:
        return data.sel(time=var.time)

@filter
def lead_time(data: xr.DataArray, var: camps.Variable) -> xr.DataArray:
    if var.lead_time is None:
        return data
    else:
        return data.sel(leadtime=var.lead_time)
