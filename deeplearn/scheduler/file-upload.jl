
const ROOT::String = "/root/data/uploads"

function mk_unique_filename(root::AbstractString, extension::AnstractString)
end

mime_extensions = Dict{AbstractString, AbstractString}(
    "image/jpeg" => "jpg",
    "image/png" => "png",
    "image/bmp" => "bmp"
)

function write_json(stream::HTTP.Stream, json_object::Dict)
    x = JSON.json(json_object)
    ℓ = length(x)
    write(stream, x)
end

function save_stream(stream::HTTP.Stream, content_type::AbstractString, content_length::Uint64, user::AbstractString)
    detected_type = HTTP.sniff(http)
    extension = mime_extensions[detected_extension]
    local_file_name = make_unique_filename(ROOT, extension)
    bytes_read = 0

    open(local_file_name, 'rb') do save
        while !eof(stream)
            bytes = readavailable(stream)
            ℓ = length(bytes)
            bytes_read += ℓ
            write(save, bytes)
        end
    end

    hash = checksum(local_file_name)
    file_stats = stat(local_file_name)
    metadata = Dict(
        :hash => hash,
        :file_name => local_file_name,
        :created_time => Dates.now(Dates.UTC),
        :created_by => user,
        :mimetype => content_type,
        :detected_mimeype => detected_type
        :filesize => filestats.size
    )
    new_id = insert!(db_connection, "vault__files", metadata)
    write_json(stream, Dict(:id => new_id, :hash => hash, :filesize => filestats.size))
end


get_routes = Dict{String, Function}()
post_routes = Dict{String, Function}()

@inline is_symbol(expr) = (typeof(expr) == Symbol)
@inline is_typed_variable(expr) = (typeof(expr) == Expr) && (expr.head === :(::))
@inline is_variable_with_default(expr) = (typeof(expr) == Expr) && (expr.head === :(=))
@inline is_variable_spec(expr) = (is_symbol(expr) || is_typed_variable(expr) || is_variable_with_default(expr))


@inline function get_variable_name(expr)
    if is_typed_variable(expr)
        expr.args[1]
    elseif is_variable_with_default(expr)
        get_variable_name(expr.args[1])
    else
        expr
    end
end


@inline function get_variable_type(expr)
    if is_typed_variable(expr)
        expr.args[2]
    elseif is_variable_with_default(expr)
        get_variable_type(expr.args[1])
    else
        :Any
    end
end


@inline function get_variable_default(expr)
    if is_variable_with_default(expr)
        expr.args[2]
    else
        nothing
    end
end


@inline function parse_parameter_tuple(args)
    query_args = Array([])
    for parameter ∈ args
        if is_variable_spec(parameter)
            name = get_variable_name(parameter)
            type_ = get_variable_type(parameter)
            default = get_variable_default(parameter)
            push!(query_args, [name, type_, default])
        else
            throw("Badd parameter syntax")
        end
    end
    return query_args
end

QueryStringValue = Union{AbstractString, Void}

Base.parse(T::Type{Y}, x::Y) where {Y} = x
Base.parse(T::Type{AbstractString}, x::QueryStringValue) = x
Base.parse(T::Type{Any}, x::QueryStringValue) = x
Base.parse(T::Type{Y}, x::Union{Y, Void}) where {Y} = x


@inline function parse_function_arguments(param)
    #query_args = param.args
    if typeof(param) == Symbol
        # parameter is a single symbol without type
       return Array([[param, :Any]])
    else
        # param is an expression
        type_ = param.head
        if type_ === Symbol("::")
            A = param.args
            return Array([[param, A[2]]])
        elseif type_ === :tuple
            return parse_parameter_tuple(param.args)
        else
            throw("Baddd parameter syntax")
        end
    end

end

@inline function make_function_body(query_args, func_name, body)
    wrapper_function = :(
        function $(esc(func_name))(stream, route, query)
        end
    )
    wrapper_function = :(
        (stream, route, headers, query) -> begin end
    )
    
    f_body = wrapper_function.args[2].args
    query_parse = Array([])
    for (arg_name, arg_type, arg_default) ∈ query_args
        println(arg_name, " ", arg_type, " ", arg_default)
        arg_key = String(arg_name)
        push!(f_body, :($arg_name = Base.parse($arg_type, get(query, $arg_key, $arg_default))))
    end
    push!(f_body, body)
    return wrapper_function
end

macro GET(route, func_name, param, body)
    query_args = parse_function_arguments(param)
    wrapper_function = make_function_body(query_args, func_name, body)
    
    :(begin
        get_routes[$route] = $wrapper_function #$(esc(func_name))
    end) 
end

macro POST(route, func_name, param, body)
    query_args = parse_function_arguments(param)
    wrapper_function = make_function_body(query_args, func_name, body)
    
    :(begin
        # $wrapper_function
        post_routes[$route] = $wrapper_function #$(esc(func_name))
    end) 
end
