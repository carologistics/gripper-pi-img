��    �        �   
      �  �  �  h   1  �   �  i  �  b  �  �   X     �       %     5   :     p     ~     �     �  "   �     �      �     �       	   /     9  "   N  4   q  *   �  .   �        
             %     7     K     ]     z     �     �     �     �     �     �          %     ;     Q     f          �     �     �     �     �     �               9     T  %   t  &   �  #   �  #   �  #   	  !   -  (   O  <   x  %   �     �      �       #   9     ]     }  #   �  %   �  ?   �  	        )  %   =     c  
   s     ~     �     �     �     �     �     �     �          !     5     C     U  4   m     �     �     �     �     �               -     E     Z     c     x     �     �  '   �     �  &   �  	   �                 #      &      /      @      E      U   L   o   ;   �      �   "   !     :!     G!     U!     b!     p!     t!  	   }!     �!  *   �!  $   �!  +   �!  #   "  7   0"  %   h"  "   �"  4   �"  (   �"  (   #     8#     H#     [#     o#     �#     �#     �#     �#     �#  
   �#  &   �#  '   $  ,   *$     W$     ]$     r$     v$      �$  �  �$  �  n&  v   )  �   �)  �  q*  ~  �+  �   {.     */     9/  $   V/  2   {/     �/     �/     �/     �/  1   �/  )   0  5   =0  .   s0     �0     �0     �0  (   �0  ;   1  *   Q1  4   |1     �1     �1     �1     �1     �1     2  =   !2     _2     w2     �2     �2     �2     �2     3     %3     B3     `3     ~3  +   �3     �3     �3  7   �3     !4     (4     44     S4     e4  $   �4  %   �4  *   �4  1   �4  9   ,5  .   f5  1   �5  .   �5  ,   �5  A   #6  H   e6  5   �6  +   �6  /   7  1   @7  0   r7  %   �7     �7  -   �7  6   8  G   L8     �8     �8  ,   �8     �8     �8     9     9     -9     K9     `9     w9     �9     �9     �9     �9     �9     �9     :  :   ,:     g:     z:     �:  "   �:     �:     �:     �:     
;     ';  	   A;      K;     l;     r;     �;  8   �;     �;  4   �;     <     <  
   *<     5<     <<     X<     n<     t<  %   �<  A   �<  4   �<     %=  .   B=     q=     ~=     �=     �=     �=     �=     �=     �=  (   �=     �=  2   >     M>  2   h>  "   �>     �>  4   �>     ?     #?     :?     J?     ^?     r?     �?     �?     �?     �?      �?     �?  1   �?  /   @  7   O@     �@     �@     �@  $   �@  (   �@     C   �   ,       '                                      r      �       �   �   T   L   |       \   %   9      e   "   �   �   H       A   �   �       �   �       �          }   j          h         +   �           #   (               �      �   �                 k   [   <   �   �       D   0   ]   x   t   w   Q   s           �   z   7   O       a   d          .           Y       {   R       5   K   y   )       /       U       c      �       !       �   �   
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
PO-Revision-Date: 2013-11-20 09:58+0000
Last-Translator: Matej Urbančič <>
Language-Team: Slovenian (http://www.transifex.com/lennart/avahi/language/sl/)
Language: sl
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
Plural-Forms: nplurals=4; plural=(n%100==1 ? 0 : n%100==2 ? 1 : n%100==3 || n%100==4 ? 2 : 3);
     -h --help            Pokaži pomoč
    -V --version         Pokaži različico
    -D --browse-domains  Prebrskaj brskalne domene namesto storitev
    -a --all             Pokaži vse storitve, ne glede na vrsto
    -d --domain=DOMENA   Domena za brskanje
    -v --verbose         Omogoči podroben način
    -t --terminate       Zaključi po izmetu bolj ali manj popolnega seznama
    -c --cache           Zaključi po izmetu vseh vnosov iz medpomnilnika
    -l --ignore-local    Prezri krajevne storitve
    -r --resolve         Razloči najdene storitve
    -f --no-fail         Ne spodleti, če ozadnji program ni na voljo
    -p --parsable        Izhod v razčlenljivi obliki
     -k --no-db-lookup    Ne poizveduj o vrstah storitve
    -b --dump-db         Izvrzi zbirko podatkov vrst storitev
 %s [možnosti]

    -h --help Pokaže to pomoč
    -s --ssh Brskanje za SSH strežniki
    -v --vnc Brskanje za VNC strežniki
    -S --shell Brskanje za SSH in VNC strežniki
    -d --domain=DOMENA Domena za brskanje
 %s [možnosti] %s <ime gostitelja ...>
%s [možnosti] %s <naslov ... >

    -h --help            Pokaži pomoč
    -V --version         Pokaži različico
    -n --name            Razloči ime gostitelja
    -a --address         Razloči naslov
    -v --verbose         Omogoči zgovoren način
    -6                   Poizvedi o naslovu IPv6
    -4                   Poizvedi o naslovu IPv4
 %s [možnosti] %s <ime> <vrsta> <vrata> [<besedilo ...>]
%s [možnosti] %s <ime-gostitelja> <naslov>

    -h --help            Pokaži pomoč
    -V --version         Pokaži različico
    -s --service         Objavi storitev
    -a --address         Objavi naslov
    -v --verbose         Omogoči podrobni način
    -d --domain=DOMENA   Domena, kjer bo objavljena storitev
    -H --host=DOMENA     Gostitelj, kjer je doma storitev
       --subtype=PODVRSTA Dodatna podvrsta za registracijo te storitve
    -R --no-reverse      Ne objavi obratnega vnosa z naslovom
    -f --no-fail         Ne spodleti, če ozadnji program ni na voljo
 %s [možnosti] <ime novega gostitelja>

    -h --help            Pokaži pomoč
    -V --version         Pokaži različico
    -v --verbose         Omogoči podrobni način
 : Vse za zdaj
 : Predpomnilnik je izčrpan
 <i>Trenutno ni izbrane storitve.</i> Nično zaključen seznam vrst storitev za brskanje Dostop je zavrnjen Naslov Družina naslova Naslov: Prišlo je do nepričakovane napake vodila D-Bus. Dejanje odjemalca Avahi je spodletelo: %s Dejanje domenskega brskalnika Avahi je spodletelo: %s Razreševanje podatkov Avahi je spodletelo: %s Napačno število argumentov.
 Slabo stanje Brskanje med vrstami storitev Seznam brskanja vrst storitev je prazen! Brskanje za storitev vrste %s v domeni %s je spodletela: %s Brskanje za storitvami v domeni <b>%s</b>: Brskanje za storitvami na <b>krajevnem omrežju</b>: Brskanje ... Preklicano.
 Spremeni domeno Izbor strežnika SSH Izbor lupinskega strežnika Izbor strežnika VNC Spodletelo izvajanje odjemalca, zato bo dejanje končano: %s
 Povezovanje z '%s' ...
 Spodletel odziv DNS: FORMERR Spodletel odziv DNS: NOTAUTH Spodletel odziv DNS: NOTIMP Spodletel odziv DNS: NOTZONE Spodletel odziv DNS: NXDOMAIN Spodletel odziv DNS: NXRRSET Spodletel odziv DNS: REFUSED Spodletel odziv DNS: SERVFAIL Spodletel odziv DNS: YXDOMAIN Spodletel odziv DNS: YXRRSET Povezava z ozadnjim programom je spodletela Ozadnji program ni zagnan Namizje Povezava je prekinjena; poteka ponovno povezovanje ...
 Domena Ime domene: E Ifce Prot %-*s %-20s domena
 Domena Ifce Prot
 Vzpostavljeno pod imenom '%s'
 Dodajanje naslova je spodletelo: %s
 Dodajanje storitve je spodletelo: %s
 Dodajanje podvrste '%s' je spodletelo: %s
 Povezovanje s strežnikom Avahi je spodletelo: %s Ustvarjanje naslovnega razreševalnika je spodletelo: %s
 Ustvarjanje brskalnika za %s je spodletelo: %s Ustvarjanje predmeta odjemalca je spodletelo: %s
 Ustvarjanje brskalnika domen je spodletelo: %s Ustvarjanje skupine vnosa je spodletelo: %s
 Ustvarjanje razreševalnika gostiteljskih imen je spodletelo: %s
 Ustvarjanje razreševalnika za %s vrste %s v domeni %s je spodletelo: %s Ustvarjanje predmeta enostavne ankete je spodletelo.
 Razčlenjevanje naslova '%s' je spodletelo
 Razreševanje številke vrat je spodletelo: %s
 Poizvedovanje imena gostitelja je spodletelo: %s
 Poizvedovanje niza različice je spodletelo: %s
 Branje domene Avahi je spodletelo: %s Vpisovanje je spodletelo: %s
 Razreševanje naslova '%s' je spodletelo: %s
 Razreševanje imena gostitelja '%s' je spodletelo: %s
 Razreševanje storitve '%s' vrste '%s' v domeni '%s' je spodletelo: %s
 Ime gostitelja Ime gostitelja je v sporu
 Ime gostitelja je uspešno spremenjeno v %s
 Začenjanje ... Vmesnik: Neveljaven DNS TTL Neveljaven razred DNS Neveljavna DNS vrnitvena koda Neveljavna vrsta DNS Neveljavna koda napake Neveljaven podatek RDATA Neveljaven naslov Neveljaven argument Neveljavna nastavitev Neveljavno ime domene Neveljavne zastavice Neveljavno ime gostitelja Neveljavno kazalo vmesnika Neveljavno število argumentov, saj je pričakovan le en.
 Neveljavno dejanje Neveljaven paket Neveljavna številka vrat Neveljavna specifikacija protokola Neveljaven zapis Neveljaven ključ zapisa Neveljavno ime storitve Neveljavna podvrsta storitve Neveljavna vrsta storitve Je prazno Krajevno poimenovanje je v sporu Mesto Pomnilnik je izčrpan Ime Poimenovanje je v sporu, zato bo izbrano novo ime '%s'.
 Ni določenega ukaza.
 Na voljo ni nobenega ustreznega omrežnega protokola Ni zadetkov Ni dovoljeno Ni podprto V redu Napaka operacijskega okolja Dejanje je spodletelo Vrata Razrešene storitve Ime gostitelja storitve razreševanja Samodejno razreši ime gostitelja izbrane storitve pred povratkom Samodejno razreši izbrano storitev pred povrnitvijo Ključ zapisa vira je vzorec Različica strežnika: %s; ime gostitelja: %s
 Ime storitve Ime storitve: Vrsta storitve Vrsta storitve: TXT Podatki TXT Podatki TXT: Terminal Številka vrat IP za razrešeno storitev Podatki TXT razrešene storitve Družina naslova za razreševanje imena gostitelja Naslov razrešene storitve Domena za brskanje oziroma NULL za privzeto domeno Ime gostitelja razrešene storitve Poslan predmet ni veljaven Zahtevana operacija je zaradi odvečnosti neveljavna Ime izbrane storitve Vrsta izbrane storitve Čas je potekel Premalo argumentov
 Preveč argumentov
 Preveč odjemalcev Preveč vnosov Preveč predmetov Vrsta Neskladje različic Čakanje na ozadnji program ...
 _Domena ... ukaz avahi_domain_browser_new() je spodletel: %s
 ukaz avahi_service_browser_new() spodletel: %s
 ukaz avahi_service_type_browser_new() je spodletel: %s
 prazno ukaz execlp() je spodletel: %s
 ni na voljo brskalnik storitev je spodletel: %s
 ukaz service_type_browser spodletel: %s
 