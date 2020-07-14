from blimpy import Waterfall
import queue, threading

"""Load parts of waterfall on a separate thread so that
main process can immediately get next queued part without
spending time loading it in."""

class ThreadedWaterfallLoader:
    def __init__(self, candidate_file, freq_windows, max_memory=1):
        self.candidate_file = candidate_file
        self.freq_windows = freq_windows
        self.max_memory = max_memory # max amount of data in GB to load into memory at once

        self.q = queue.Queue(maxsize=1) # hold next Waterfall observation to be processed
        self.q_freqs = queue.Queue() # make Queue to put in freq windows

        self.thread = threading.Thread(target=self.load_data_from_file, daemon=True)
        self.finished = False

    def start(self):
        # make Queue to put in freq windows
        for freq_pair in self.freq_windows:
            self.q_freqs.put(freq_pair)

        self.thread.start() # begin looping self.load_data_from_file() until done

    def load_data_from_file(self):
        while not self.finished: # exit if finished is set to True
            if self.q.empty():
                try: # attempt to get pair of freqs
                    f_start, f_stop = self.q_freqs.get(block=False)
                except queue.Empty:
                    print("No more frequency pairs to load from freqs queue. Exiting thread.")
                    self.finished = True

                wt_obs = Waterfall(self.candidate_file, f_start=f_start, f_stop=f_stop, max_load=self.max_memory/2)
                freqs, data = wt_obs.grab_data(f_start, f_stop) # get candidate freqs and data
                wt_obs = None # de-reference object to free up memory
                # place data into the queue, ready to be processed
                self.q.put((freqs, data))

    def get_observation(self):
        if self.finished: # no more freqs to process and no data left to unpack
            raise ValueError("No more observations in queue")

        # take observation from queue
        obs_loaded = self.q.get()
        self.q.task_done()

        return obs_loaded

    def stop(self):
        self.finished = True # breaks out of loop in self.load_data_from_file()
        self.thread.join()