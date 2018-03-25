


mutable struct MonitoredSocket <: TCPSocket
    channel::IO
    bytes_read::UInt64
    bytes_written::UInt64
end

MonitoredSocket(io::IO) = MonitoredSocket(io, 0, 0)

function Base.readavailable(stream::MonitoredSocket)
    bytes = readavailable(stream.channel)
    byted_read += length(bytes)
    return bytes
end

Base.eof(stream::MonitoredSocket) = Base.eof(stream.channel)
Base.close(stream::MonitoredSocket) = Base.close(stream.channel)
Base.isopen(stream::MonitoredSocket) = Base.isopen(stream.channel)
Base.bytesavailable(stream::MonitoredSocket) = Base.bytesavailable(stream.channel)

function Base.unsafe_write(stream::MonitoredSocket, p::Ptr{UInt8}, n::UInt)
    written = unsafe_write(stream.channel, p, n)
    stream.bytes_written += written
    return written
end

