mergeInto(LibraryManager.library, {
  OneBeyondEvent: function (message) {
    var canvas = Module['canvas'];
    canvas.dispatchEvent(new CustomEvent('one-beyond-unity', {detail: UTF8ToString(message)}));
  }
});
