if {![package vsatisfies [package provide Tcl] 8.6.0]} return
package ifneeded Tk 8.6.13 [list load [file normalize [file join /usr/lib/aarch64-linux-gnu libtk8.6.so]]]
