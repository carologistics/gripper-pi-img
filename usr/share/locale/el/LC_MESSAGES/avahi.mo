��    �        �   
      �  �  �  h   1  �   �  i  �  b  �  �   X     �       %     5   :     p     ~     �     �  "   �     �      �     �       	   /     9  "   N  4   q  *   �  .   �        
             %     7     K     ]     z     �     �     �     �     �     �          %     ;     Q     f          �     �     �     �     �     �               9     T  %   t  &   �  #   �  #   �  #   	  !   -  (   O  <   x  %   �     �      �       #   9     ]     }  #   �  %   �  ?   �  	        )  %   =     c  
   s     ~     �     �     �     �     �     �     �          !     5     C     U  4   m     �     �     �     �     �               -     E     Z     c     x     �     �  '   �     �  &   �  	   �                 #      &      /      @      E      U   L   o   ;   �      �   "   !     :!     G!     U!     b!     p!     t!  	   }!     �!  *   �!  $   �!  +   �!  #   "  7   0"  %   h"  "   �"  4   �"  (   �"  (   #     8#     H#     [#     o#     �#     �#     �#     �#     �#  
   �#  &   �#  '   $  ,   *$     W$     ]$     r$     v$      �$  �  �$  �  V&  �   *+  m  �+  @  ]-    �/  �   �3     �4  2   �4  4   �4  h   "5  #   �5     �5  +   �5     �5  =   6  '   @6  8   h6  )   �6  -   �6     �6  2   7  _   H7  f   �7  K   8  P   [8     �8     �8     �8  )   �8  6   9  )   R9  0   |9     �9     �9     �9     :     $:     B:     a:     :     �:     �:     �:  9   �:  /   3;  #   c;  B   �;     �;     �;  $   �;     <  0   -<  =   ^<  ;   �<  >   �<  N   =  P   f=  H   �=  R    >  I   S>  P   �>  c   �>  u   R?  j   �?  <   3@  @   p@  R   �@  T   A  8   YA  &   �A  @   �A  S   �A  n   NB     �B  ;   �B  S   C     mC     �C     �C     �C  2   �C     �C  .   D     DD  $   UD     zD     �D  *   �D     �D  6   �D  ,   *E  V   WE  &   �E     �E  -   �E  8   F  $   TF  5   yF  /   �F  0   �F  1   G     BG  2   VG     �G  $   �G  
   �G  T   �G  &   !H  L   HH     �H     �H  !   �H     �H     �H  &   I     /I  !   8I  W   ZI  �   �I  s   UJ  E   �J  O   K     _K     }K     �K     �K     �K     �K     �K     L  N   L  H   mL  s   �L  D   *M  f   oM  U   �M  L   ,N  `   yN  T   �N  R   /O  $   �O     �O     �O  ,   �O  6   P  6   CP  
   zP  !   �P  "   �P     �P  0   �P  1   Q  6   >Q     uQ     ~Q     �Q  %   �Q  *   �Q     C   �   ,       '                                      r      �       �   �   T   L   |       \   %   9      e   "   �   �   H       A   �   �       �   �       �          }   j          h         +   �           #   (               �      �   �                 k   [   <   �   �       D   0   ]   x   t   w   Q   s           �   z   7   O       a   d          .           Y       {   R       5   K   y   )       /       U       c      �       !       �   �   
   l   ;      v              �          3   �   p   _       n         u      E       S   W   8   M   N   ^       1      I   *   F   o           b   �   @   g              �       �   q           2   6   V          X   �      ?           &       P   �          i   	             -      =       $       �           m   �   �   J          G      �      >   f   ~           4   B           :   Z   `        -h --help            Show this help
    -V --version         Show version
    -D --browse-domains  Browse for browsing domains instead of services
    -a --all             Show all services, regardless of the type
    -d --domain=DOMAIN   The domain to browse in
    -v --verbose         Enable verbose mode
    -t --terminate       Terminate after dumping a more or less complete list
    -c --cache           Terminate after dumping all entries from the cache
    -l --ignore-local    Ignore local services
    -r --resolve         Resolve services found
    -f --no-fail         Don't fail if the daemon is not available
    -p --parsable        Output in parsable format
     -k --no-db-lookup    Don't lookup service types
    -b --dump-db         Dump service type database
 %s [options]

    -h --help            Show this help
    -s --ssh             Browse SSH servers
    -v --vnc             Browse VNC servers
    -S --shell           Browse both SSH and VNC
    -d --domain=DOMAIN   The domain to browse in
 %s [options] %s <host name ...>
%s [options] %s <address ... >

    -h --help            Show this help
    -V --version         Show version
    -n --name            Resolve host name
    -a --address         Resolve address
    -v --verbose         Enable verbose mode
    -6                   Lookup IPv6 address
    -4                   Lookup IPv4 address
 %s [options] %s <name> <type> <port> [<txt ...>]
%s [options] %s <host-name> <address>

    -h --help            Show this help
    -V --version         Show version
    -s --service         Publish service
    -a --address         Publish address
    -v --verbose         Enable verbose mode
    -d --domain=DOMAIN   Domain to publish service in
    -H --host=DOMAIN     Host where service resides
       --subtype=SUBTYPE An additional subtype to register this service with
    -R --no-reverse      Do not publish reverse entry with address
    -f --no-fail         Don't fail if the daemon is not available
 %s [options] <new host name>

    -h --help            Show this help
    -V --version         Show version
    -v --verbose         Enable verbose mode
 : All for now
 : Cache exhausted
 <i>No service currently selected.</i> A NULL terminated list of service types to browse for Access denied Address Address family Address: An unexpected D-Bus error occurred Avahi client failure: %s Avahi domain browser failure: %s Avahi resolver failure: %s Bad number of arguments
 Bad state Browse Service Types Browse service type list is empty! Browsing for service type %s in domain %s failed: %s Browsing for services in domain <b>%s</b>: Browsing for services on <b>local network</b>: Browsing... Canceled.
 Change domain Choose SSH server Choose Shell Server Choose VNC server Client failure, exiting: %s
 Connecting to '%s' ...
 DNS failure: FORMERR DNS failure: NOTAUTH DNS failure: NOTIMP DNS failure: NOTZONE DNS failure: NXDOMAIN DNS failure: NXRRSET DNS failure: REFUSED DNS failure: SERVFAIL DNS failure: YXDOMAIN DNS failure: YXRRSET Daemon connection failed Daemon not running Desktop Disconnected, reconnecting ...
 Domain Domain Name: E Ifce Prot %-*s %-20s Domain
 E Ifce Prot Domain
 Established under name '%s'
 Failed to add address: %s
 Failed to add service: %s
 Failed to add subtype '%s': %s
 Failed to connect to Avahi server: %s Failed to create address resolver: %s
 Failed to create browser for %s: %s Failed to create client object: %s
 Failed to create domain browser: %s Failed to create entry group: %s
 Failed to create host name resolver: %s
 Failed to create resolver for %s of type %s in domain %s: %s Failed to create simple poll object.
 Failed to parse address '%s'
 Failed to parse port number: %s
 Failed to query host name: %s
 Failed to query version string: %s
 Failed to read Avahi domain: %s Failed to register: %s
 Failed to resolve address '%s': %s
 Failed to resolve host name '%s': %s
 Failed to resolve service '%s' of type '%s' in domain '%s': %s
 Host Name Host name conflict
 Host name successfully changed to %s
 Initializing... Interface: Invalid DNS TTL Invalid DNS class Invalid DNS return code Invalid DNS type Invalid Error Code Invalid RDATA Invalid address Invalid argument Invalid configuration Invalid domain name Invalid flags Invalid host name Invalid interface index Invalid number of arguments, expecting exactly one.
 Invalid operation Invalid packet Invalid port number Invalid protocol specification Invalid record Invalid record key Invalid service name Invalid service subtype Invalid service type Is empty Local name collision Location Memory exhausted Name Name collision, picking new name '%s'.
 No command specified.
 No suitable network protocol available Not found Not permitted Not supported OK OS Error Operation failed Port Resolve Service Resolve Service Host Name Resolve the host name of the selected service automatically before returning Resolve the selected service automatically before returning Resource record key is pattern Server version: %s; Host name: %s
 Service Name Service Name: Service Type Service Type: TXT TXT Data TXT Data: Terminal The IP port number of the resolved service The TXT data of the resolved service The address family for host name resolution The address of the resolved service The domain to browse in, or NULL for the default domain The host name of the resolved service The object passed in was not valid The requested operation is invalid because redundant The service name of the selected service The service type of the selected service Timeout reached Too few arguments
 Too many arguments
 Too many clients Too many entries Too many objects Type Version mismatch Waiting for daemon ...
 _Domain... avahi_domain_browser_new() failed: %s
 avahi_service_browser_new() failed: %s
 avahi_service_type_browser_new() failed: %s
 empty execlp() failed: %s
 n/a service_browser failed: %s
 service_type_browser failed: %s
 Project-Id-Version: Avahi
Report-Msgid-Bugs-To: https://github.com/lathiat/avahi/issues
PO-Revision-Date: 2014-05-05 03:28+0000
Last-Translator: ΔΗΜΗΤΡΗΣ ΣΠΙΓΓΟΣ <dmtrs32@gmail.com>
Language-Team: Greek (http://www.transifex.com/lennart/avahi/language/el/)
Language: el
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
Plural-Forms: nplurals=2; plural=(n != 1);
     -h --help            Εμφάνιση αυτής της βοήθειας
    -V --version         Εμφάνιση έκδοσης
    -D --browse-domains  Περιήγηση για τομείς περιήγησης αντί για υπηρεσίες
    -a --all             Εμφάνιση όλων των υπηρεσιών, ανεξάρτητα από τον τύπο
    -d --domain=ΤΟΜΕΑΣ   Ο τομέας για περιήγηση
    -v --verbose         Ενεργοποίηση αναλυτικής κατάστασης
    -t --terminate       Τερματισμός μετά την αποτύπωση ενός περισσότερο ή λιγότερο πλήρους καταλόγου
    -c --cache           Τερματισμός μετά την αποτύπωση όλων των καταχωρίσεων από την κρυφή μνήμη
    -l --ignore-local    Παράβλεψη τοπικών υπηρεσιών
    -r --resolve         Επίλυση των υπηρεσιών που βρέθηκαν
    -f --no-fail         Να μην αποτυγχάνει αν ο δαίμονας δεν είναι διαθέσιμος
    -p --parsable        Έξοδος σε αναλύσιμη μορφή
     -k --no-db-lookup    Να μην αναζητούνται τύποι υπηρεσιών
    -b --dump-db         Αποτύπωση βάσης δεδομένων τύπου υπηρεσιών
 %s [options]

    -h --help            Εμφάνιση της βοήθειας
    -s --ssh             Περιήγηση εξυπηρετητών SSH
    -v --vnc             Περιήγηση εξυπηρετητών VNC
    -S --shell           Περιήγηση και SSH και VNC
    -d --domain=ΤΟΜΕΑΣ   Ο τομέας για περιήγηση
 %s [options] %s <host name ...>
%s [options] %s <address ... >

    -h --help            Εμφάνιση αυτής τηςβοήθειας
    -V --version         Εμφάνιση έκδοσης
    -n --name            Μετάφραση ονόματος υπολογιστή
    -a --address         Μετάφραση διεύθυνσης
    -v --verbose         Ενεργοποίηση αναλυτικής κατάστασης
    -6                   Αναζήτηση διεύθυνσης IPv6
    -4                   Αναζήτηση διεύθυνσης IPv4
 %s [options] %s <name> <type> <port> [<txt ...>]
%s [options] %s <host-name> <address>

    -h --help            Εμφάνιση αυτής της βοήθειας
    -V --version         Εμφάνιση έκδοσης
    -s --service         Δημοσίευση υπηρεσίας
    -a --address         Δημοσίευση διεύθυνσης
    -v --verbose         Ενεργοποίηση αναλυτικής κατάστασης
    -d --domain=ΤΟΜΕΑΣ   Ο τομέας για δημοσίευση υπηρεσίας
    -H --host=ΤΟΜΕΑΣ     Ο οικοδεσπότης όπου βρίσκεται η υπηρεσία
       --subtype=ΥΠΟΤΥΠΟΣ Ένας πρόσθετος υποτύπος για καταχώριση αυτής της υπηρεσίας
    -R --no-reverse      Να μην δημοσιευτεί αντίστροφη καταχώριση με διεύθυνση
    -f --no-fail         Να μην αποτυγχάνει αν ο δαίμονας  δεν είναι διαθέσιμος
 %s [options] <new host name>

    -h --help            Εμφάνιση βοήθειας
    -V --version         Εμφάνιση έκδοσης
    -v --verbose         Ενεργοποίηση αναλυτικής κατάστασης
 : Όλα για τώρα
 : Η κρυφή μνήμη εξαντλήθηκε
 <i>Δεν επιλέχθηκε υπηρεσία.</i> Ένα NULL τερμάτισε τη λίστα τύπων υπηρεσιών για εξερεύνηση Αποτυχία πρόσβασης Διεύθυνση Οικογένεια διευθύνσεων Διεύθυνση: Συνέβη ένα απροσδόκητο σφάλμα D-Bus Αποτυχία πελάτη Avahi: %s Αποτυχία περιηγητή τομέα Avahi: %s Αποτυχία επιλύτη Avahi: %s Κακός αριθμός ορισμάτων
 Κακή κατάσταση Εξερεύνηση τύπων υπηρεσιών Ο κατάλογος τύπων υπηρεσιών περιήγησης είναι κενός! Αποτυχία περιήγησης για τύπο υπηρεσίας %s στον τομέα %s: %s Περιήγηση για υπηρεσίες στον τομέα <b>%s</b>: Εξερεύνηση υπηρεσιών στο <b>τοπικό δίκτυο</b>: Εξερεύνηση... Ακυρώθηκε.
 Αλλαγή τομέα Επιλογή εξυπηρετητή SSH Επιλογή εξυπηρετητή κελύφους Επιλογή εξυπηρετητή VNC Αποτυχία πελάτη, έξοδος: %s
 Σύνδεση με '%s' ...
 Αποτυχία DNS: FORMERR Αποτυχία DNS: NOTAUTH Αποτυχία DNS: NOTIMP Αποτυχία DNS: NOTZONE Αποτυχία DNS: NXDOMAIN Αποτυχία DNS: NXRRSET Αποτυχία DNS: REFUSED Αποτυχία DNS: SERVFAIL Αποτυχία DNS: YXDOMAIN Αποτυχία DNS: YXRRSET Η σύνδεση με το δαίμονα απέτυχε Ο δαίμονας δεν εκτελείται Επιφάνεια εργασίας Αποσύνδεση, γίνεται επανασύνδεση ...
 Τομέας Όνομα τομέα: Τομέας E Ifce Prot %-*s %-20s
 Τομέας E Ifce Prot
 Δημιουργήθηκε με όνομα '%s'
 Αδυναμία προσθήκης διεύθυνσης: %s
 Αδυναμία προσθήκης υπηρεσίας: %s
 Αποτυχία προσθήκης υποτύπου '%s': %s
 Αδυναμία σύνδεσης με τον εξυπηρετητή Avahi: %s Αποτυχία δημιουργίας επιλύτη διεύθυνσης: %s
 Αδυναμία δημιουργίας περιηγητή για %s: %s Αδυναμία δημιουργίας αντικειμένου πελάτη: %s
 Αποτυχία δημιουργίας περιηγητή τομέα: %s Αποτυχία δημιουργίας ομάδας καταχώρισης: %s
 Αποτυχία δημιουργίας επιλύτη ονόματος οικοδεσπότη: %s
 Αποτυχία δημιουργίας επιλύτη για το %s του τύπου %s στον τομέα %s: %s Αποτυχία δημιουργίας απλού αντικειμένου σταθμοσκόπησης.
 Αποτυχία ανάλυσης διεύθυνσης '%s'
 Αποτυχία ανάλυσης αριθμού θύρας: %s
 Αποτυχία ερωτήματος ονόματος οικοδεσπότη: %s
 Αποτυχία ερωτήματος συμβολοσειράς έκδοσης: %s
 Αποτυχία ανάγνωσης τομέα Avahi: %s Αποτυχία εγγραφής: %s
 Αποτυχία επίλυσης διεύθυνσης '%s': %s
 Αποτυχία επίλυσης ονόματος οικοδεσπότη '%s': %s
 Αποτυχία επίλυσης υπηρεσίας '%s' του τύπου '%s' στον τομέα '%s': %s
 Όνομα υπολογιστή Σύγκρουση ονομάτων υπολογιστών
 Επιτυχής αλλαγή του ονόματος υπολογιστή σε %s
 Αρχικοποίηση... Διεπαφή: Άκυρο DNS TTL Άκυρη κλάση DNS Άκυρος κώδικα επιστροφής DNS Άκυρος τύπος DNS Άκυρος κώδικας σφάλματος Άκυρο RDATA Μη έγκυρη διεύθυνση Άκυρο όρισμα Άκυρη ρύθμιση Λανθασμένο όνομα τομέα Άκυρες σημαίες Λανθασμένο όνομα εξυπηρετητή Άκυρος δείκτης διεπαφής Άκυρος αριθμός ορισμάτων, αναμένεται μόνο ένα.
 Μη έγκυρη λειτουργία Άκυρο πακέτο Μη έγκυρος αριθμός θύρας Άκυρη προδιαγραφή πρωτοκόλλου Μη έγκυρη καταγραφή Μη έγκυρη καταγραφή κλειδιού Μη έγκυρο όνομα υπηρεσίας Άκυρος υποτύπος υπηρεσίας Μη έγκυρος τύπος υπηρεσίας Είναι κενό Σύγκρουση τοπικού ονόματος Τοποθεσία Η μνήμη εξαντλήθηκε Όνομα Σύγκρουση ονομάτων, επιλογή νέου ονόματος '%s'.
 Δεν ορίστηκε εντολή.
 Δεν υπάρχει διαθέσιμο πρωτόκολλο δικτύου Δε βρέθηκε Δεν επιτρέπεται Δεν υποστηρίζεται Εντάξει Σφάλμα OS Η λειτουργία απέτυχε Θύρα Επίλυση υπηρεσίας Επίλυση του ονόματος οικοδεσπότη της υπηρεσίας Επίλυση του ονόματος οικοδεσπότη της επιλεγμένης υπηρεσίας αυτόματα πριν την επιστροφή Επίλυση της επιλεγμένης υπηρεσίας αυτόματα πριν την επιστροφή Το κλειδί εγγραφής πόρου είναι μοτίβο Έκδοση εξυπηρετητή: %s; Όνομα οικοδεσπότη: %s
 Όνομα Υπηρεσίας Όνομα υπηρεσίας: Τύπος υπηρεσίας Τύπος υπηρεσίας: TXT Δεδομένα TXT Δεδομένα ΤΧΤ: Τερματικό Ο αριθμός θύρας IP της επιλυμένης υπηρεσίας Τα δεδομένα TXT της επιλυμένης υπηρεσίας Η οικογένεια διευθύνσεων για την ανάλυση ονόματος οικοδεσπότη Η διεύθυνση της επιλυμένης υπηρεσίας Ο τομέας για περιήγηση, ή NULL για τον προεπιλεγμένο τομέα Το όνομα οικοδεσπότη της επιλυμένης υπηρεσίας Το αντικείμενο που πέρασε δεν ήταν έγκυρο Η ζητούμενη λειτουργία είναι άκυρη επειδή πλεονάζει Το όνομα υπηρεσίας για την επιλεγμένη συσκευή Ο τύπος υπηρεσίας για την επιλεγμένη συσκευή Λήξη χρονικού ορίου Λίγα ορίσματα
 Πολλά ορίσματα
 Μεγάλος αριθμός πελατών Μεγάλος αριθμός καταχωρήσεων Μεγάλος αριθμός αντικειμένων Τύπος Ασυμφωνία έκδοσης Αναμονή δαίμονα ...
 _Τομέας... Αποτυχία avahi_domain_browser_new(): %s
 Αποτυχία avahi_service_browser_new(): %s
 Αποτυχία avahi_service_type_browser_new(): %s
 κενό execlp() απέτυχε: %s
 μη διαθέσιμο Αποτυχία service_browser: %s
 Αποτυχία service_type_browser: %s
 