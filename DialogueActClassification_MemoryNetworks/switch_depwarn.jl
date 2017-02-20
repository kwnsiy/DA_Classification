@enum DepwarnFlag DepwarnOff=0 DepwarnOn=1 DepwarnError=2

doc"""
- `switch_depwarn!(flag :: Bool)`
- `switch_depwarn!(flag :: DepwarnFlag)`
Enable/Disable deprecation warning. 
- `DepwarnOff` or `false` : switch off deprecation warning
- `DepwarnOn` or `true` : switch on deprecation warning
- `DepwarnError` : turn deprecation warning into error
"""
switch_depwarn!(flag :: Bool) = switch_depwarn!(flag ? DepwarnOn : DepwarnOff)
function switch_depwarn!(flag :: DepwarnFlag)
    old_opt = Base.JLOptions()
    params = map(fieldnames(Base.JLOptions)) do name
        name == :depwarn ? Int(flag) : getfield(old_opt, name)
    end
    new_opt = Base.JLOptions(params...)
    unsafe_store!(cglobal(:jl_options, Base.JLOptions), new_opt)
    flag
end

# one-liner
# unsafe_store!(cglobal(:jl_options, Base.JLOptions), Base.JLOptions(map(fieldnames(Base.JLOptions)) do name; name==:depwarn?0:getfield(Base.JLOptions(), name) end...))
