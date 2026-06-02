(define (problem picture_193)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump blue_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_pump)
        (clear blue_battery)
        (part_at blue_battery table)
        (part_at red_pump pump_placement)
    )
    
    (:goal
        (and
            (part_at blue_battery battery_placement)
        )
    )
)