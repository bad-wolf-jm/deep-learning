
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
