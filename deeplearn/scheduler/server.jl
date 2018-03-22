module Servers

function do_handle_connection(f::Function, c::Connection)
    try
        handle_connection(f, io) # ; kw...)
    catch e
        @error "Error:   $io" e catch_stacktrace()
    finally
        close(io)
        @info "Closed:  $io"
    end
end

function listen(f::Function, host::String="127.0.0.1", port::Int=8081)
    Log.info("Listening on: $host:$port")
    tcpserver = Sockets.listen(Sockets.getaddrinfo(host), port)

    tcpref[] = tcpserver

    try
        while isopen(tcpserver)
            try
                io = accept(tcpserver)
            catch e
                if e isa Base.UVError
                    Log.warning("$e")
                    break
                else
                    rethrow(e)
                end
            end
            let io = Connection(host, string(port), pipeline_limit, 0, io)
                Log.info("Accept:  $io")
                @async do_handle_connection(f, io)
            end
        end
    catch e
        if typeof(e) <: InterruptException
            Log.warning("Interrupted: listen($host,$port)")
        else
            rethrow(e)
        end
    finally
        close(tcpserver)
    end

    return
end


"""
Start a timeout monitor task to close the `Connection` if it is inactive.
Create a `Transaction` object for each HTTP Request received.
"""
function handle_connection(f::Function, c::Connection)
    # ;
    #                        reuse_limit::Int=nolimit,
    #                        readtimeout::Int=0, kw...)

    wait_for_timeout = Ref{Bool}(true)
    if readtimeout > 0
        @async while wait_for_timeout[]
            Log.info("$(inactiveseconds(c))")
            if inactiveseconds(c) > readtimeout
                Log.warning("Timeout: $c")
                writeheaders(c.io, Response(408, ["Connection" => "close"]))
                close(c)
                break
            end
            sleep(8 + rand() * 4)
        end
    end

    try
        while isopen(c)
            io = Transaction(c)
            handle_transaction(f, io)
        end
    finally
        wait_for_timeout[] = false
    end
    return
end


"""
Create a `HTTP.Stream` and parse the Request headers from a `HTTP.Transaction`.
If there is a parse error, send an error Response.
Otherwise, execute stream processing function `f`.
"""
function handle_transaction(f::Function, t::Transaction;
                            final_transaction::Bool=false,
                            verbose::Bool=false, kw...)

    request = HTTP.Request()
    http = Streams.Stream(request, t)

    try
        startread(http)
    catch e
        if e isa EOFError && isempty(request.method)
            return
        elseif e isa HTTP.ParseError
            @error e
            status = e.code == :HEADER_SIZE_EXCEEDS_LIMIT  ? 413 : 400
            write(t, Response(status, body = string(e.code)))
            close(t)
            return
        else
            rethrow(e)
        end
    end

    if verbose
        @info http.message
    end

    response = request.response
    response.status = 200
    if final_transaction || hasheader(request, "Connection", "close")
        setheader(response, "Connection" => "close")
    end

    @async try
        handle_stream(f, http)
    catch e
        if isioerror(e)
            @warn e
        else
            @error e catch_stacktrace()
        end
        close(t)
    end
    return
end

"""
Execute stream processing function `f`.
If there is an error and the stream is still open,
send a 500 response with the error message.
Close the `Stream` for read and write (in case `f` has not already done so).
"""
function handle_stream(f::Function, http::Stream)

    try
            f(http)
    catch e
        if isopen(http) && !iswritable(http)
            @error e catch_stacktrace()
            http.message.response.status = 500
            startwrite(http)
            write(http, sprint(showerror, e))
        else
            rethrow(e)
        end
    end

    closeread(http)
    closewrite(http)
    return
end


end # module