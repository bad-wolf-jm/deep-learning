
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

macro get (route, func_name, params...)
end

macro GET(route, func_name, param, body)
    #query_args = param.args
    if typeof(param) == Symbol
        # parameter is a single symbol without type
        query_args = Array([[param, :Any]])
    else
        # param is an expression
        type_ = param.head
        if type_ === Symbol("::")
            A = param.args
            query_args =  Array([[param, A[2]]])
        elseif type_ === :tuple
            query_args = Array([])
            for parameter ∈ param.args
                if typeof(parameter) == Symbol
                    push!(query_args, [parameter, :Any])
                elseif typeof(parameter) == Expr
                    type_ = parameter.head
                    if type_ === Symbol("::")
                        A = parameter.args
                        push!(query_args, [A[1], A[2]])
                    else
                        println("BADD Syntax")
                    end
                else
                    println("BAR SYNTAX")
                end
            end
            
        end
    end
    
    wrapper_function = :(
        function $(esc(func_name))(stream, query)
        end
    )
    
    f_body = wrapper_function.args[2].args
    
    println(query_args)
    query_parse = Array([])
    for (arg_name, arg_type) ∈ query_args
        push!(f_body, :($arg_name = parse($arg_type, query[$arg_name])))
    end
    push!(f_body, body)
    
    return :(begin
            $wrapper_function
            routes[$route] = $(esc(func_name))
            end) 
end
