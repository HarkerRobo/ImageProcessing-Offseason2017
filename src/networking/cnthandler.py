"""
Because clients may want to choose to not receive contours, this module defines a handler that
allows clients to be picky.
"""

from . import messages as m

def create_cnt_handler(handler):
    """Creates a handler to handle contour subscription messages.

    This acts as middleware between another handler, which you will have to pass in. It will return
    a dynamic set of clients subscibed to contour messages and a handler in a tuple.
    """

    receive_set = set()

    def handle_bb_message(client):
        """Subscribe a client to contour messages."""
        receive_set.add(client)

    def handle_message(client, message_str):
        """Handle a message sent to the socket.

        This function will be returned by surrounding function.
        """
        try:
            message = m.parse_message(message_str)
            if message[m.FIELD_TYPE] == m.TYPE_REQUEST_CNT:
                handle_bb_message(client)
            else:
                handler(client, message_str) # Pass the message on
        except ValueError as e:
            msg = m.create_message(m.TYPE_ERROR, {m.FIELD_ERROR: str(e)})
            try:
                client.send(msg.encode('utf-8'))
            except:
                pass

    return receive_set, handle_message
