from twisted.web import server, resource
from twisted.internet import reactor, endpoints, task

i=0

class Simple(resource.Resource):
    isLeaf = True
    def render_GET(self, request):
        return b"<html>Hello, world!</html>" + str(i).encode('utf8')

def call_loop():
    global i
    i += 1

#foo = task.LoopingCall(call_loop)
#foo.start(0)

site = server.Site(Simple())
endpoint = endpoints.TCP4ServerEndpoint(reactor, 8080)
endpoint.listen(site)
reactor.run()
